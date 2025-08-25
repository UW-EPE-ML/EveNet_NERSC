import argparse
import yaml
import os
import sys
# Add the parent directory of the notebook to the path
sys.path.append(os.path.abspath("../.."))
import h5py

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


from preprocessing.preprocess import unflatten_dict
from functools import partial
import glob
import pandas as pd
from rich.table import Table
from rich.console import Console

def read_from_files(files):
    # Example: Read all Parquet files in a folder
    file_list = glob.glob(files)

    # Read and concatenate
    df = pd.concat([pd.read_parquet(f) for f in file_list], ignore_index=True)
    batch = {col: df[col].to_numpy() for col in df.columns}
    return batch

def main(args):
    indir = args.in_dir
    sample = read_from_files(f"{indir}/*parquet")
    shape_metadata = json.load(open(f"{indir}/shape_metadata.json"))
    normalization_file = torch.load(f"{indir}/normalization.pt")

    event_info_file = args.event_info_file
    with open(event_info_file) as f:
        config = yaml.safe_load(f)

    df = process_event_batch(
        sample,
        shape_metadata=shape_metadata,
        unflatten=unflatten_dict,
    )

    # Convert the DataFrame to a dictionary format suitable for SPANet

    x = dict()
    iseq = 0
    for items in config["INPUTS"]["SEQUENTIAL"]:
        x[f"INPUTS/{items}/MASK"] = df['x_mask']
        for element in config["INPUTS"]["SEQUENTIAL"][items]:
            x[f"INPUTS/{items}/{element}"] = df['x'][..., iseq]
            iseq += 1
    iglo = 0
    for items in config["INPUTS"]["GLOBAL"]:
        for element in config["INPUTS"]["GLOBAL"][items]:
            x[f"INPUTS/{items}/{element}"] = df['conditions'][..., iglo]
            iglo += 1
    for items in config['CLASSIFICATIONS']["EVENT"]:
        x[f"CLASSIFICATIONS/EVENT/{items}"] = df['classification']

    imother = 0

    for mother, decays in (config['EVENT'][next(iter(config['EVENT']))]).items():
        idaughter = 0
        for daughter in decays:

            x[f"TARGETS/{mother}/{daughter}"] = df['assignments-indices'][:, imother, idaughter]
            idaughter += 1
        imother += 1

    x["INFO/EVENT_WEIGHT"] = df['event_weight']

    os.makedirs(args.store_dir, exist_ok=True)
    output_file = f"{args.store_dir}/data.h5"
    data = x
    # Save to HDF5
    with h5py.File(output_file, 'w') as h5f:
        for key, value in data.items():
            # Create groups if they don't exist
            group_path = '/'.join(key.split('/')[:-1])
            dataset_name = key.split('/')[-1]
            grp = h5f.require_group(group_path)
            # Write dataset
            grp.create_dataset(dataset_name, data=value)

if __name__ == '__main__':
    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('event_info_file', help='Path to config file', default=None)
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--store_dir', type=str, default='Storage')
    args = parser.parse_args()

    main(args)