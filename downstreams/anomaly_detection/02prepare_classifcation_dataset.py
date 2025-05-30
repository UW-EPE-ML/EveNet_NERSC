import argparse
import yaml
import os

import json
import numpy as np
import pyarrow.parquet as pq
import awkward as ak
import vector
vector.register_awkward()
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from evenet.control.global_config import global_config
from evenet.dataset.preprocess import process_event_batch, convert_batch_to_torch_tensor
from evenet.network.evenet_model import EveNetModel
from evenet.utilities.diffusion_sampler import DDIMSampler


from helpers.basic_fit import plot_mass_distribution
from helpers.flow_sampling import get_mass_samples
from helpers.stats_functions import curve_fit_m_inv, parametric_fit, check_bkg_for_peaks
from helpers.plotting import read_feature
from helpers.utils import save_file, save_df

from rich.progress import Progress, BarColumn, TimeRemainingColumn

from preprocessing.preprocess import unflatten_dict
from functools import partial
import glob
import pandas as pd



def read_from_files(files):
    # Example: Read all Parquet files in a folder
    file_list = glob.glob(files)

    # Read and concatenate
    df = pd.concat([pd.read_parquet(f) for f in file_list], ignore_index=True)
    batch = {col: df[col].to_numpy() for col in df.columns}
    return batch

def clean_and_append(dirname, postfix):
    if dirname.endswith("/"):
        dirname = dirname[:-1]
    return dirname + postfix

def mix(args):

    postfix = "" if not args.no_signal else "_no_signal"
    with open(args.config_workflow) as f:
        config = yaml.safe_load(f)
    step_dir = f"step2_mix_SR{postfix}"
    inputdir = clean_and_append(config["output"]["storedir"], postfix)

    os.makedirs(os.path.join(config["output"]["plotdir"], step_dir), exist_ok=True)
    global_config.load_yaml(config["train"]["config"])
    normalization_dict = torch.load(os.path.join(inputdir, "SB", "normalization.pt"))
    shape_metadata = json.load(open(os.path.join(inputdir, "SB", "shape_metadata.json")))
    with open(config["input"]["event_info"]) as f:
        event_info = yaml.safe_load(f)


    df_dummy_SB = pq.read_table(
        os.path.join(inputdir, "SB", "data.parquet")
    ).to_pandas()
    df_dummy_SB.sample(frac=1).reset_index(drop=True)
    df_dummy_SR = pq.read_table(
        os.path.join(inputdir, "SR", "data.parquet")
    ).to_pandas()
    df_dummy_SR.sample(frac=1).reset_index(drop=True)
    norm_dict = torch.load(os.path.join(inputdir, "SB", "normalization.pt"))

    df_SB = process_event_batch(
        df_dummy_SB,
        shape_metadata=shape_metadata,
        unflatten=unflatten_dict,
    )

    df_SR = process_event_batch(
        df_dummy_SR,
        shape_metadata=shape_metadata,
        unflatten=unflatten_dict,
    )

    inv_mass_truth = np.concatenate([df_SB["conditions"][..., 0], df_SR["conditions"][..., 0]], axis=0)

    inv_mass_results = plot_mass_distribution(
        inv_mass_truth,
        SR_left=config['mass-windows']['SR-left'],
        SR_right=config['mass-windows']['SR-right'],
        SB_left=config['mass-windows']['SB-left'],
        SB_right=config['mass-windows']['SB-right'],
        bkg_fit_degree=config['fit']['bkg-fit-degree'],
        num_bins_SR=config['mass-windows']['SR-bins'],
        save_name=os.path.join(config['output']['plotdir'], step_dir, "truth_mass_distribution.png")
    )

    # Check for peaks in the background
    gen_sample = read_from_files(os.path.join(clean_and_append(config["output"]["storedir"], f"_gen{postfix}"), args.region, "data*.parquet"))
    gen_sample = process_event_batch(
        gen_sample,
        shape_metadata=shape_metadata,
        unflatten=unflatten_dict,
    )

#    jet = ak.from_regular(vector.zip(
#        {
#            "pt": ak.from_numpy(read_feature(gen_sample["x"], event_info, 'pt')),
#            "eta": ak.from_numpy(read_feature(gen_sample["x"], event_info, 'eta')),
#            "phi": ak.from_numpy(read_feature(gen_sample["x"], event_info, 'phi')),
#            "mass": ak.from_numpy(read_feature(gen_sample["x"], event_info, 'mass')),
#            "MASK": ak.from_numpy(gen_sample['x_mask'])
#        }
#    ))

#    inv_mass_gen = (jet[..., 0] + jet[..., 1]).mass
#    _ = plot_mass_distribution(
#        inv_mass_gen,
#        SR_left=config['mass-windows']['SR-left'],
#        SR_right=config['mass-windows']['SR-right'],
#        SB_left=config['mass-windows']['SB-left'],
#        SB_right=config['mass-windows']['SB-right'],
#        bkg_fit_degree=config['fit']['bkg-fit-degree'],
#        num_bins_SR=config['mass-windows']['SR-bins'],
#        save_name=os.path.join(config['output']['plotdir'], step_dir, f"gen_mass_distribution_{args.region}.png")
#    )
#
#    gen_sample["conditions"] = inv_mass_gen.to_numpy().reshape(-1, 1)
#
#    SR_filter = (inv_mass_gen.to_numpy() > config['mass-windows']['SR-left']) & (
#                inv_mass_gen.to_numpy() < config['mass-windows']['SR-right'])
#    SB_filter = (inv_mass_gen.to_numpy() > config['mass-windows']['SB-left']) & (
#                inv_mass_gen.to_numpy() < config['mass-windows']['SB-right'])
#    SB_filter = SB_filter & ~SR_filter
#    if args.region == "SR":
#        gen_sample = {k: v[SR_filter] for k, v in gen_sample.items()}
#    else:
#        gen_sample = {k: v[SB_filter] for k, v in gen_sample.items()}


    if args.region == "SR":
        df_data = df_SR
    else:
        df_data = df_SB

    gen_sample["classification"] = np.zeros_like(gen_sample["classification"], dtype=np.int64)
    df_data["classification"] = np.ones_like(df_data["classification"], dtype=np.int64)


    print(gen_sample["x"].shape[0])

    print(gen_sample["x"].shape, df_data["x"].shape)

    # Expand the mask to broadcast over the last dimension (N)
    mask_expanded = df_data['x_mask'][:, :, np.newaxis]  # (B, P, 1)
    # Apply the mask: set unmasked values to +inf so they don't affect min
    x_masked = np.where(mask_expanded, df_data['x'], np.inf)  # (B, P, N)
    # Get minimum over both batch (B) and position (P) axes
    min_vals = np.min(x_masked, axis=(0, 1))  # shape: (N,)
    x = gen_sample["x"]  # shape (M, P, N)
    x_mask = gen_sample["x_mask"]  # shape (M, P), boolean
    min_vals = min_vals  # shape (N,)
    # Expand mask to shape (M, P, 1) for broadcasting
    mask_exp = x_mask[:, :, np.newaxis]  # (M, P, 1)
    # Compare x > min_vals, broadcasted over N
    comparison = x[..., global_config.event_info.generation_pc_indices] > min_vals[global_config.event_info.generation_pc_indices]  # (M, P, N)
    # Apply the mask: only evaluate where x_mask is True
    valid = comparison & mask_exp  # (M, P, N)
    # Each masked (m, p) must satisfy all features > min_vals
    valid_particles = np.all(valid | ~mask_exp, axis=2)  # (M, P)
    # Each event: all masked particles must be valid
    keep_mask = np.all(valid_particles | ~x_mask, axis=1)  # (M,)
    # Filtered result
    gen_sample = {k: v[keep_mask] for k, v in gen_sample.items()}

    print("after x_mask filtering", gen_sample["x"].shape[0])


    min_vals_cond = np.min(df_data["conditions"], axis=(0))  # shape: (N,)
    # Apply the same logic to conditions
    comparison = gen_sample["conditions"] > min_vals_cond  # (M, N)
    print(f"Minimum values for conditions: {min_vals_cond}")
    print(gen_sample["conditions"])
    print(comparison)
    # Apply the mask: only evaluate where x_mask is True
    keep_mask = np.all(comparison[..., 1:], axis=1)  # (M,)
    print(keep_mask)
    # Filtered result
    gen_sample = {k: v[keep_mask] for k, v in gen_sample.items()}

    if args.max_background is not None and args.max_background < gen_sample["x"].shape[0]:
        gen_sample = {k: v[:args.max_background] for k, v in gen_sample.items()}

    num_gen = gen_sample["x"].shape[0]
    num_data = df_data["x"].shape[0]
    num_total = num_gen + num_data
    norm_dict['class_counts'] = torch.tensor([num_gen, num_data], dtype=torch.float32)
    norm_dict['class_balance'] = torch.tensor([num_total/num_gen, num_total/num_data], dtype=torch.float32)
    norm_dict['subprocess_counts'] = norm_dict['class_counts']
    norm_dict['subprocess_balance'] = norm_dict['class_balance']

    print("gen", gen_sample)
    print("data", df_data)

    df_hybrid = {k: np.concatenate([gen_sample[k], df_data[k]], axis=0) for k in gen_sample}

    perm = np.random.permutation(len(next(iter(df_hybrid.values()))))
    df_hybrid = {k: v[perm] for k, v in df_hybrid.items()}

    outputdir = clean_and_append(config["output"]["storedir"], "_hybrid_raw")
    outputdir = clean_and_append(outputdir, postfix)

    save_file(
        save_dir = os.path.join(outputdir, f"{args.region}"),
        data_df = df_hybrid,
        norm_dict = norm_dict,
        event_filter = None,
    )

    save_df(
        save_dir=os.path.join(outputdir, f"{args.region}"),
        data_df=df_hybrid,
        pc_index=dict(zip(global_config.event_info.generation_pc_names, global_config.event_info.generation_pc_indices)),
        global_index=dict(zip(global_config.event_info.generation_pc_condition_names, global_config.event_info.generation_pc_condition_indices))
    )

    save_df(
        save_dir=os.path.join(outputdir, "SB"),
        data_df=df_SB,
        pc_index=dict(zip(global_config.event_info.generation_pc_names, global_config.event_info.generation_pc_indices)),
        global_index=dict(zip(global_config.event_info.generation_pc_condition_names, global_config.event_info.generation_pc_condition_indices)),
    )



    outputdir = clean_and_append(config["output"]["storedir"], "_hybrid")
    outputdir = clean_and_append(outputdir, postfix)


    df_x_mask = np.zeros_like(df_hybrid["x"])
    df_x_mask[..., global_config.event_info.generation_pc_indices] = 1.0
    df_hybrid["x"] = df_hybrid["x"] * df_x_mask
    df_hybrid["conditions"][..., 0]  = 0 # Remove the mass condition
    save_file(
        save_dir = os.path.join(outputdir, f"{args.region}"),
        data_df = df_hybrid,
        norm_dict = norm_dict,
        event_filter = None,
    )


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type = str)
    parser.add_argument("--region", type = str, default = "SR")
    parser.add_argument("--no_signal", action = "store_true", default = False)
    parser.add_argument("--max_background", type = int, default = None)
    # Parse command-line arguments
    args = parser.parse_args()
    # Explore the provided HDF5 file

    mix(args)
if __name__ == "__main__":
    main()
