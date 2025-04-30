import pandas as pd
import glob
import yaml, json
import torch
import os, sys
import awkward as ak
import vector
vector.register_awkward()
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import argparse
from preprocessing.preprocess import unflatten_dict, flatten_dict
from evenet.dataset.preprocess import process_event_batch
from helpers.plotting import *
from helpers.physics_functions import get_bins
from helpers.stats_functions import parametric_fit, curve_fit_m_inv, calculate_test_statistic
from helpers.data_transforms import logit_transform
import pyarrow.parquet as pq

def clean_and_append(dirname, postfix):
    if dirname.endswith("/"):
        dirname = dirname[:-1]
    return dirname + postfix

def mean_std_last_dim(x, x_mask):
    """
    Compute masked mean and std over all axes except the last one.
    Supports x of shape (a, c) or (a, b, c), and x_mask of shape (a,) or (a, b).

    Parameters:
        x (np.ndarray): Input array of shape (..., c)
        x_mask (np.ndarray): Boolean mask matching all dims except the last

    Returns:
        tuple of np.ndarray: (mean, std) with shape (1, c)
    """
    # Get shape info
    if x_mask.ndim == x.ndim - 1:

        # Broadcast mask to match x shape
        mask_expanded = np.expand_dims(x_mask, axis=-1)  # shape (..., 1)
        mask_broadcasted = np.broadcast_to(mask_expanded, x.shape)
    else:
        mask_broadcasted = x_mask

    # Create masked array
    x_masked = np.ma.masked_array(x, mask=~mask_broadcasted)

    # Compute mean and std over all axes except the last one
    axis_to_reduce = tuple(range(x.ndim - 1))
    mean = x_masked.mean(axis=axis_to_reduce, keepdims=True)
    std = x_masked.std(axis=axis_to_reduce, keepdims=True)

    return mean.filled(np.nan), std.filled(np.nan)

def plot_mass_distribution(
    inv_mass,
    SR_left,
    SR_right,
    SB_left,
    SB_right,
    bkg_fit_degree = 5,
    num_bins_SR = 20,
    save_name = None
):

    plot_bins_all, plot_bins_SR, plot_bins_left, plot_bins_right, plot_centers_all, plot_centers_SR, plot_centers_SB = get_bins(SR_left, SR_right, SB_left, SB_right, num_bins_SR= num_bins_SR)

    x = np.linspace(SB_left, SB_right, 100) # plot curve fit

    plt.figure(figsize = (10,5))

    # curve fit the data
    popt_0, _, _, _, _ = curve_fit_m_inv(inv_mass.to_numpy(), bkg_fit_degree, SR_left, SR_right, plot_bins_left, plot_bins_right, plot_centers_SB)
    # plot the best fit curve
    # cßalculate the test statistic
    plt.plot(x, parametric_fit(x, *popt_0), lw = 3, linestyle = "dashed")
    #S, B, q0 = calculate_test_statistic(inv_mass.to_numpy(), SR_left, SR_right, SB_left, SB_right, num_bins_SR, degree = bkg_fit_degree, starting_guess = popt_0)
    q0 = 0
    # plot all data
    plt.hist(inv_mass.to_numpy(), bins = plot_bins_all, lw = 2, histtype = "step", density = False, label = f"sig: {round(np.sqrt(q0),3)}")

    plt.axvline(SR_left)
    plt.axvline(SR_right)

    plt.xlabel("Dijet M [GeV]")
    plt.ylabel("Counts")
    plt.legend(loc = "upper right")

    if save_name is not None:
        plt.savefig(save_name)
    plt.show()

def read_from_files(files):
    # Example: Read all Parquet files in a folder
    file_list = glob.glob(files)

    # Read and concatenate
    df = pd.concat([pd.read_parquet(f) for f in file_list], ignore_index=True)
    batch = {col: df[col].to_numpy() for col in df.columns}
    return batch

def pad_object(obj, nMax):
  pad_awkward = ak.pad_none(obj, target = nMax, clip = True)
  return pad_awkward

def read_feature(x, event_info, feature):
    index = list(event_info['INPUTS']['SEQUENTIAL']['Source']).index(feature)
    normal = event_info['INPUTS']['SEQUENTIAL']['Source'][feature]
    input_x = x[..., index]
    if "log" in normal.lower():
        input_x = np.expm1(input_x)
    return input_x

def fill_value(array, event_info, input_awkarray):
    for feature_index, feature in enumerate(list(event_info['INPUTS']['SEQUENTIAL']['Source'])):
        feature_value = ak.to_numpy(getattr(input_awkarray, feature))
        if "log" in event_info['INPUTS']['SEQUENTIAL']['Source'][feature]:
            feature_value = np.log1p(np.clip(feature_value, 1e-10, None))
        array[..., feature_index] = feature_value
    return array

def save_file(
    save_dir,
    data_df,
    norm_dict,
    event_filter
):

    os.makedirs(save_dir, exist_ok=True)
    filtered_df = {col: data_df[col][event_filter.to_numpy()] for col in data_df}
    flatten_data, meta_data = flatten_dict(filtered_df)
    ### Save to parquet
    pq.write_table(flatten_data, f"{save_dir}/data.parquet")

    with open(f"{save_dir}/shape_metadata.json", "w") as f:
        json.dump(meta_data, f)

    print(f"[INFO] Final table size: {flatten_data.nbytes / 1024 / 1024:.2f} MB")
    print(f"[Saving] Saving {flatten_data.num_rows} rows to {save_dir}/data.parquet")

    torch.save(norm_dict, f"{save_dir}/normalization.pt")


def produce_dataset(args):

    postfix = "" if not args.no_signal else "_no_signal"
    step_dir = f"step0_skim{postfix}"

    with open(args.config_workflow) as f:
        config = yaml.safe_load(f)
    # Read the input files
    signal = read_from_files(config["input"]["signal"]["file"])
    background = read_from_files(config["input"]["background"]["file"])
    signal_shape = json.load(open(config["input"]["signal"]["shape"]))
    background_shape = json.load(open(config["input"]["background"]["shape"]))
    with open(config["input"]["event_info"]) as f:
        event_info = yaml.safe_load(f)

    signal_df = process_event_batch(
        signal,
        shape_metadata=signal_shape,
        unflatten=unflatten_dict
    )
    background_df = process_event_batch(
        background,
        shape_metadata=background_shape,
        unflatten=unflatten_dict
    )

    signal_number = signal_df['x'].shape[0]
    background_number = background_df['x'].shape[0]
    if background_number / signal_number > config["input"]["max-bkg-sig-ratio"]:
        background_df = {
            key: background_df[key][:int(signal_number * config["input"]["max-bkg-sig-ratio"])] for key in background_df
        }
        print(f"Reduce background to {background_df['x'].shape[0]} events")

    print(f"use signal {signal_df['x'].shape[0]} and background {background_df['x'].shape[0]} events")

    if args.no_signal:
        data_df = background_df
    else:
        data_df = {
            key: np.concatenate([signal_df[key], background_df[key]]) for key in signal_df
        }
    if config["input"]["shuffle"]:
        perm = np.random.permutation(len(next(iter(data_df.values()))))
        data_df = {k: v[perm] for k, v in data_df.items()}

    jet = ak.from_regular(vector.zip(
        {
            "pt": ak.from_numpy(read_feature(data_df['x'], event_info, 'pt')),
            "eta": ak.from_numpy(read_feature(data_df['x'], event_info, 'eta')),
            "phi": ak.from_numpy(read_feature(data_df['x'], event_info, 'phi')),
            "mass": ak.from_numpy(read_feature(data_df['x'], event_info, 'mass')),
            "MASK": ak.from_numpy(data_df['x_mask'])
        }
    ))
    jet = jet[ak.argsort(jet.pt, ascending=False)]

    jet = jet[jet.MASK]
    jet_filter = ak.ones_like(jet.pt, dtype = bool)

    # Apply jet selection
    for feature in config['jet-selection']:
        for selection_type in config['jet-selection'][feature]:
            if selection_type == 'min':
                jet_filter = jet_filter & (getattr(jet, feature) >= config['jet-selection'][feature][selection_type])
            elif selection_type == 'max':
                jet_filter = jet_filter & (getattr(jet, feature)  <= config['jet-selection'][feature][selection_type])
            elif selection_type == 'exact':
                jet_filter = jet_filter & (getattr(jet, feature)  == config['jet-selection'][feature][selection_type])
            else:
                raise ValueError(f"Unknown selection type: {selection_type}")

    # Apply event filter
    jet = jet[jet_filter]
    njet = ak.num(jet, axis = -1)
    event_filter = ak.ones_like(njet, dtype = bool)
    for selection_type in config['njet-selection']:
        if selection_type == 'min':
            event_filter = event_filter & (njet >= config['njet-selection'][selection_type])
        elif selection_type == 'max':
            event_filter = event_filter & (njet <= config['njet-selection'][selection_type])
        elif selection_type == 'exact':
            event_filter = event_filter & (njet == config['njet-selection'][selection_type])
        else:
            raise ValueError(f"Unknown selection type: {selection_type}")

    jet = jet[event_filter]
    data_df = {col: data_df[col][event_filter.to_numpy()] for col in data_df}
    inv_mass = (jet[...,0] + jet[...,1]).mass

    # Mass wondows cut
    mass_windows_filter = (inv_mass < config['mass-windows']['SB-right']) & (inv_mass > config['mass-windows']['SB-left'])
    inv_mass = inv_mass[mass_windows_filter]
    jet = jet[mass_windows_filter]
    data_df = {col: data_df[col][mass_windows_filter.to_numpy()] for col in data_df}


    storedir = clean_and_append(config['output']['storedir'], postfix)
    os.makedirs(os.path.join(config['output']['plotdir'], step_dir), exist_ok=True)
    os.makedirs(storedir, exist_ok=True)

    plot_mass_distribution(
        inv_mass = inv_mass,
        SR_left = config['mass-windows']['SR-left'],
        SR_right = config['mass-windows']['SR-right'],
        SB_left = config['mass-windows']['SB-left'],
        SB_right = config['mass-windows']['SB-right'],
        bkg_fit_degree = config['fit']['bkg-fit-degree'],
        num_bins_SR = config['mass-windows']['SR-bins'],
        save_name = os.path.join(config['output']['plotdir'], step_dir, "mass_distribution.png")
    )

    jet = pad_object(jet, config['padding']['pad-number'])
    batch_size, nobj, nfeature = data_df['x'].shape

    # Assignm jet back to data
    data_df['x_mask'] = ak.to_numpy(jet.MASK)
    data_df['x'] = np.zeros((batch_size, config['padding']['pad-number'], nfeature), dtype = np.float32)
    data_df['x'] = fill_value(data_df['x'], event_info, jet)
    data_df['conditions'] = np.expand_dims(ak.to_numpy(inv_mass), axis = 1)
    data_df['num_sequential_vectors'] = ak.to_numpy(ak.num(jet))
    data_df['num_vectors'] = data_df['num_sequential_vectors'] + 1

    mean_x, std_x = mean_std_last_dim(data_df['x'], data_df['x_mask'])
    mean_cond, std_cond = mean_std_last_dim(data_df['conditions'], data_df['conditions_mask'])

    norm_dict = dict()
    norm_dict['input_mean'] = {'Source': torch.tensor(mean_x.flatten(), dtype=torch.float32), 'Conditions': torch.tensor(mean_cond.flatten(), dtype=torch.float32)}
    norm_dict['input_std'] = {'Source': torch.tensor(std_x.flatten(), dtype=torch.float32), 'Conditions': torch.tensor(std_cond.flatten(), dtype=torch.float32)}
    norm_dict['input_num_mean'] = {"Source": torch.tensor([0.0], dtype=torch.float32)}
    norm_dict['input_num_std'] = {"Source": torch.tensor([1.0], dtype=torch.float32)}
    norm_dict['regression_mean'] = {}
    norm_dict['regression_std'] = {}
    norm_dict['class_counts'] = torch.tensor([data_df['x'].shape[0]], dtype=torch.float32)
    norm_dict['class_balance'] = torch.tensor([1], dtype=torch.float32)
    norm_dict['particle_balance'] = {}
    norm_dict['invisible_mean'] = {"Source": norm_dict['input_mean']['Source']}
    norm_dict['invisible_std'] =  {"Source": norm_dict['input_std']['Source']}
    norm_dict['subprocess_counts'] = norm_dict['class_counts']
    norm_dict['subprocess_balance'] = norm_dict['class_balance']

    SBL_filter = (inv_mass < config['mass-windows']['SR-left']) & (inv_mass > config['mass-windows']['SB-left'])
    SBR_filter = (inv_mass < config['mass-windows']['SB-right']) & (inv_mass > config['mass-windows']['SR-right'])
    SR_filter = (inv_mass < config['mass-windows']['SR-right']) & (inv_mass > config['mass-windows']['SR-left'])
    SB_filter = SBL_filter | SBR_filter

    save_file(
        save_dir = os.path.join(storedir, "SB"),
        data_df = data_df,
        norm_dict = norm_dict,
        event_filter = SB_filter
    )

    save_file(
        save_dir = os.path.join(storedir, "SR"),
        data_df = data_df,
        norm_dict = norm_dict,
        event_filter = SR_filter
    )

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type = str)
    parser.add_argument("--no_signal", action="store_true", help="Skip signal processing")
    # Parse command-line arguments
    args = parser.parse_args()

    # Explore the provided HDF5 file
    produce_dataset(args)

if __name__ == "__main__":
    main()

