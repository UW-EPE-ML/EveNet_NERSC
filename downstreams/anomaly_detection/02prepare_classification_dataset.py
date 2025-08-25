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
from rich.pretty import pprint

from evenet.control.global_config import global_config
from evenet.dataset.preprocess import process_event_batch

from helpers.plotting import read_feature
from helpers.utils import save_file, save_df

from preprocessing.preprocess import unflatten_dict
from functools import partial
import glob
import pandas as pd
from rich.table import Table
from rich.console import Console

from jetnet.evaluation import cov_mmd


def reprocess_sample(x, event_info, feature_names):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    stack_tensor = []
    for feature in feature_names:
        stack_tensor.append(read_feature(x, event_info, feature))

    pc = torch.stack(stack_tensor, dim=-1)  # shape: (B, P, N)
    return pc



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


    gen_performance = dict()
    postfix = "" if not args.no_signal else "_no_signal"
    with open(args.config_workflow) as f:
        config = yaml.safe_load(f)
    step_dir = f"step2_mix_SR{postfix}"
    inputdir = clean_and_append(config["output"]["storedir"], postfix)

    os.makedirs(os.path.join(config["output"]["plotdir"], step_dir), exist_ok=True)

    cwd = os.getcwd()
    base_dir = os.path.dirname(os.path.abspath(args.config_workflow))
    os.chdir(base_dir)
    global_config.load_yaml(config["train"]["config"])
    normalization_dict = torch.load(os.path.join(inputdir, "SB", "normalization.pt"))
    shape_metadata = json.load(open(os.path.join(inputdir, "SB", "shape_metadata.json")))
    with open(config["input"]["event_info"]) as f:
        event_info = yaml.safe_load(f)

    os.chdir(cwd)

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

    # inv_mass_truth = np.concatenate([df_SB["conditions"][..., 0], df_SR["conditions"][..., 0]], axis=0)
    #
    # inv_mass_results = plot_mass_distribution(
    #     inv_mass_truth,
    #     SR_left=config['mass-windows']['SR-left'],
    #     SR_right=config['mass-windows']['SR-right'],
    #     SB_left=config['mass-windows']['SB-left'],
    #     SB_right=config['mass-windows']['SB-right'],
    #     bkg_fit_degree=config['fit']['bkg-fit-degree'],
    #     num_bins_SR=config['mass-windows']['SR-bins'],
    #     save_name=os.path.join(config['output']['plotdir'], step_dir, "truth_mass_distribution.png")
    # )

    # Check for peaks in the background
    gen_sample = read_from_files(os.path.join(clean_and_append(config["output"]["storedir"], f"_gen{postfix}"), args.region, "data*.parquet"))
    gen_sample = process_event_batch(
        gen_sample,
        shape_metadata=shape_metadata,
        unflatten=unflatten_dict,
    )

    # Reoder the generated sample based on pt

    print("before sorting", gen_sample["x"][0])
    pt_gen = read_feature(gen_sample["x"], event_info, 'pt')
    sorted_indices = np.argsort(-pt_gen, axis=1)  # shape: (B, P)
    B, P, N = gen_sample["x"].shape
    batch_indices = np.arange(B)[:, None]  # shape: (B, 1)
    gen_sample["x"] = gen_sample["x"][batch_indices, sorted_indices]
    print("after sorting", gen_sample["x"][0])

    muon_mass = 0.1056583745  # GeV/c^2
    muon_mass_array = np.ones_like(read_feature(gen_sample["x"], event_info, 'pt')) * muon_mass

    jet = ak.from_regular(vector.zip({
           "pt": ak.from_numpy(read_feature(gen_sample["x"], event_info, 'pt')),
           "eta": ak.from_numpy(read_feature(gen_sample["x"], event_info, 'eta')),
           "phi": ak.from_numpy(read_feature(gen_sample["x"], event_info, 'phi')),
           "mass": ak.from_numpy(muon_mass_array),
           "MASK": ak.from_numpy(gen_sample['x_mask'])
    }))

    inv_mass_gen = (jet[..., 0] + jet[..., 1]).mass



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


    for cond_index, cond_name in enumerate(global_config.event_info.generation_pc_condition_names):
        if cond_name == "HT":
            print("HT condition found")
            df_SR["HT-pc"] = df_SR["conditions"][..., cond_index]
            df_SB["HT-pc"] = df_SB["conditions"][..., cond_index]
            gen_sample["HT-pc"] = (jet[..., 0].pt + jet[..., 1].pt).to_numpy()
            gen_sample["conditions"][..., cond_index] = gen_sample["HT-pc"]  # Keep the original HT for consistency
        elif cond_name == "deltaR":
            print("deltaR condition found")
            df_SR["deltaR-pc"] = df_SR["conditions"][..., cond_index]
            df_SB["deltaR-pc"] = df_SB["conditions"][..., cond_index]
            gen_sample["deltaR-pc"] = (jet[..., 0].deltaR(jet[..., 1])).to_numpy()
            gen_sample["conditions"][..., cond_index] = gen_sample["deltaR-pc"]  # Keep the original deltaR for consistency
        else:
            print(f"Condition {cond_name} not HT or deltaR, skipping")

    # Define some pc quantities
    jet_data_SB = ak.from_regular(vector.zip({
        "pt": ak.from_numpy(read_feature(df_SB["x"], event_info, 'pt')),
        "eta": ak.from_numpy(read_feature(df_SB["x"], event_info, 'eta')),
        "phi": ak.from_numpy(read_feature(df_SB["x"], event_info, 'phi')),
        "mass": np.ones_like(read_feature(df_SB["x"], event_info, 'pt')) * muon_mass,
        "MASK": ak.from_numpy(df_SB['x_mask'])
    }))
    pt_balance_data_SB = abs((jet_data_SB[..., 0] + jet_data_SB[..., 1]).pt) / (abs(jet_data_SB[..., 0].pt) + abs(
        jet_data_SB[..., 1].pt))  # Normalize the pt of the leading and subleading particles
    df_SB["pt-balance-pc"] = pt_balance_data_SB.to_numpy()

    jet_data_SR = ak.from_regular(vector.zip({
        "pt": ak.from_numpy(read_feature(df_SR["x"], event_info, 'pt')),
        "eta": ak.from_numpy(read_feature(df_SR["x"], event_info, 'eta')),
        "phi": ak.from_numpy(read_feature(df_SR["x"], event_info, 'phi')),
        "mass": np.ones_like(read_feature(df_SR["x"], event_info, 'pt')) * muon_mass,
        "MASK": ak.from_numpy(df_SR['x_mask'])
    }))
    pt_balance_data_SR = abs((jet_data_SR[..., 0] + jet_data_SR[..., 1]).pt) / (abs(jet_data_SR[..., 0].pt) + abs(
        jet_data_SR[..., 1].pt))  # Normalize the pt of the leading and subleading particles
    df_SR["pt-balance-pc"] = pt_balance_data_SR.to_numpy()

    if args.region == "SR":
        df_data = df_SR
    else:
        df_data = df_SB


    data_benchmark = reprocess_sample(df_data['x'], event_info, ['eta', 'phi', 'pt', 'ip3d'])
    gen_benchmark = reprocess_sample(gen_sample['x'], event_info, ['eta', 'phi', 'pt', 'ip3d'])

    std_data = torch.std(data_benchmark, dim=(0,1), unbiased=False).unsqueeze(0).unsqueeze(0)
    std_data = std_data + 1e-8

    data_benchmark_norm = data_benchmark / std_data

    gen_benchmark_norm = gen_benchmark / std_data

    cov, mmd = cov_mmd(
        real_jets = data_benchmark_norm,
        gen_jets = gen_benchmark_norm,
        num_eval_samples = 1000,
    )
    gen_num = gen_sample["x"].shape[0]

    gen_performance["before cut"] = {
        "cov": cov,
        "mmd": mmd,
        "efficiency": gen_sample["x"].shape[0] / gen_num
    }



    pt_balance_gen = abs((jet[..., 0] + jet[..., 1]).pt) / (abs(jet[..., 0].pt) + abs(jet[..., 1].pt))  # Normalize the pt of the leading and subleading particles
    gen_sample["pt-balance-pc"] = pt_balance_gen.to_numpy()


    SR_filter = (inv_mass_gen.to_numpy() > config['mass-windows']['SR-left']) & (
               inv_mass_gen.to_numpy() < config['mass-windows']['SR-right'])
    SB_filter = (inv_mass_gen.to_numpy() > config['mass-windows']['SB-left']) & (
               inv_mass_gen.to_numpy() < config['mass-windows']['SB-right'])
    SB_filter = SB_filter & ~SR_filter
    if args.region == "SR":
        gen_sample = {k: v[SR_filter] for k, v in gen_sample.items()}
    else:
        gen_sample = {k: v[SB_filter] for k, v in gen_sample.items()}


    gen_sample["classification"] = np.zeros_like(gen_sample["classification"], dtype=np.int64)
    df_data["classification"] = np.ones_like(df_data["classification"], dtype=np.int64)


    print(gen_sample["x"].shape[0])

    print(gen_sample["x"].shape, df_data["x"].shape)


    # # 1. Filter the generated sample based on pt of leading and subleading particles
    # pt_gen = read_feature(gen_sample["x"], event_info, 'pt')
    # pt_data = read_feature(df_data["x"], event_info, 'pt')
    #
    # # 2. Get minimum pt of leading and subleading particles across all events
    # min_leading_pt = np.min(pt_data[:, 0])
    # min_subleading_pt = np.min(pt_data[:, 1])
    #
    # pt_mask = (pt_gen[:, 0] > min_leading_pt) & (pt_gen[:, 1] > min_subleading_pt)
    # gen_sample = {k: v[pt_mask] for k, v in gen_sample.items()}

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

    print(gen_sample["conditions"])
    comparison = gen_sample["conditions"][..., global_config.event_info.generation_target_indices] > min_vals_cond[..., global_config.event_info.generation_target_indices]  # (M, N)

    print(f"Minimum values for conditions: {min_vals_cond}")
    # Apply the mask: only evaluate where x_mask is True
    keep_mask = np.all(comparison, axis=1)  # (M,)
    # Filtered result
    gen_sample = {k: v[keep_mask] for k, v in gen_sample.items()}

    gen_benchmark = reprocess_sample(gen_sample['x'], event_info, ['eta', 'phi', 'pt', 'ip3d'])
    gen_benchmark_norm = gen_benchmark / std_data
    cov, mmd = cov_mmd(
        real_jets=data_benchmark_norm,
        gen_jets=gen_benchmark_norm,
        num_eval_samples=1000,
    )
    gen_performance["after cut"] = {
        "cov": cov,
        "mmd": mmd,
        "efficiency": gen_sample["x"].shape[0] / gen_num
    }


    if args.max_background is not None and args.max_background < gen_sample["x"].shape[0]:
        gen_sample = {k: v[:args.max_background] for k, v in gen_sample.items()}

    num_gen = gen_sample["x"].shape[0]
    num_data = df_data["x"].shape[0]
    num_total = num_gen + num_data
    norm_dict['class_counts'] = torch.tensor([num_gen, num_data], dtype=torch.float32)
    norm_dict['class_balance'] = torch.tensor([num_total/num_gen, num_total/num_data], dtype=torch.float32)
    norm_dict['subprocess_counts'] = norm_dict['class_counts']
    norm_dict['subprocess_balance'] = norm_dict['class_balance']

    # gen_sample["x"][..., :-1] = np.zeros_like(gen_sample["x"][..., :-1])  # Set all but the first particle to zero
    # df_data["x"][..., :-1] = np.zeros_like(df_data["x"][..., :-1])
    #

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

    norm_cond_mask = torch.zeros_like(norm_dict["input_mean"]["Conditions"])
    norm_cond_mask[..., global_config.event_info.generation_target_indices] = 1.0
    norm_dict["input_mean"]["Conditions"] = norm_dict["input_mean"]["Conditions"] * norm_cond_mask
    norm_dict["input_std"]["Conditions"] = norm_dict["input_std"]["Conditions"] * norm_cond_mask + (1 - norm_cond_mask)

    df_hybrid["x"] = df_hybrid["x"] * df_x_mask
    cond_mask = np.zeros_like(df_hybrid["conditions"])
    cond_mask[..., global_config.event_info.generation_target_indices] = 1.0
    df_hybrid["conditions"] = df_hybrid["conditions"] * cond_mask # Remove the mass condition

    islepton_index = list(event_info['INPUTS']['SEQUENTIAL']['Source']).index("isLepton") if "isLepton" in event_info['INPUTS']['SEQUENTIAL']['Source'] else None
    if islepton_index is not None:
        df_hybrid["x"][..., islepton_index] = 1
    charge_index = list(event_info['INPUTS']['SEQUENTIAL']['Source']).index("charge") if "charge" in event_info['INPUTS']['SEQUENTIAL']['Source'] else None
    if charge_index is not None:
        df_hybrid["x"][:,0, charge_index] = -1  # Set charge to 0 for all particles
        df_hybrid["x"][:, 1, charge_index] = 1  # Set charge to 0 for all particles

    df_hybrid["conditions"] = df_hybrid["conditions"][..., global_config.event_info.generation_target_indices]
    norm_dict["input_mean"]["Conditions"] = norm_dict["input_mean"]["Conditions"][..., global_config.event_info.generation_target_indices]
    norm_dict["input_std"]["Conditions"] = norm_dict["input_std"]["Conditions"][..., global_config.event_info.generation_target_indices]



    save_file(
        save_dir = os.path.join(outputdir, f"{args.region}"),
        data_df = df_hybrid,
        norm_dict = norm_dict,
        event_filter = None,
    )

    console = Console()
    table = Table(title="Gen Performance Comparison")

    table.add_column("Metric", style="bold cyan")
    table.add_column("Before Cut", style="yellow")
    table.add_column("After Cut", style="green")

    # Collect all metric keys
    metrics = sorted(set(gen_performance["before cut"]) | set(gen_performance["after cut"]))

    for metric in metrics:
        before = gen_performance["before cut"].get(metric, "N/A")
        after = gen_performance["after cut"].get(metric, "N/A")

        # Format floats to 4 decimal places
        before_str = f"{before:.4f}" if isinstance(before, (int, float)) else str(before)
        after_str = f"{after:.4f}" if isinstance(after, (int, float)) else str(after)

        table.add_row(metric, before_str, after_str)

    console.print(table)


    with open(os.path.join(outputdir, f"{args.region}", "gen_performance.json"), "w") as f:
        json.dump(gen_performance,f, indent=4)


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
