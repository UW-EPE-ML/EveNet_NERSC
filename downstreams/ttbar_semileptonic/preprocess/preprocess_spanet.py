import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from multiprocessing import Pool, cpu_count
from collections import defaultdict

from evenet.control.global_config import global_config
from evenet.control.event_info import EventInfo
from preprocessing.monitor_gen_matching import monitor_gen_matching
from preprocessing.evenet_data_converter import EveNetDataConverter
from preprocessing.postprocessor import PostProcessor
import h5py, glob


def list_h5_files(indir):
    # Option 1: Using glob (recommended)
    return sorted(glob.glob(os.path.join(indir, "*.h5")))

    # Option 2: Using os.listdir (if you prefer)
    # return sorted([os.path.join(indir, f) for f in os.listdir(indir) if f.endswith(".h5")])


def load_all_datasets(filepath):
    """Recursively load datasets from a file into a flat dict with full paths as keys."""
    data = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            data[name] = obj[()]

    with h5py.File(filepath, 'r') as f:
        f.visititems(visit)
    return data


def concatenate_h5_datasets(indir):
    """Concatenate datasets from multiple h5 files with nested structure."""
    all_data = defaultdict(list)
    filepaths = list_h5_files(indir)

    for path in filepaths:
        file_data = load_all_datasets(path)
        for k, v in file_data.items():
            all_data[k].append(v)

    # Concatenate along axis 0
    concatenated_data = {k: np.concatenate(vs, axis=0) for k, vs in all_data.items()}

    # Change name
    prefix_replacements = {
        "INPUTS/Met/": "INPUTS/Conditions/",
        "INPUTS/Momenta/": "INPUTS/Source/"
    }

    for k in list(concatenated_data.keys()):
        for prefix, replacement in prefix_replacements.items():
            if k.startswith(prefix):
                new_key = k.replace(prefix, replacement)
                concatenated_data[new_key] = concatenated_data.pop(k)

    return concatenated_data


def generate_assignment_names(event_info: EventInfo):
    assignment_names = []
    assignment_map = []

    for p, c in event_info.product_particles.items():
        for pp, dp in c.items():
            assignment_names.append(f"TARGETS/{p}/{pp}")
            assignment_map.append((p, pp, dp))

    return assignment_names, assignment_map


def generate_regression_names(event_info: EventInfo):
    regression_keys = []
    regression_key_map = []

    common_process = set(event_info.regressions)
    # Collect all possible regression keys
    for process in sorted(common_process):
        if process != "EVENT": continue
        for regression in event_info.regressions[process]:
            regression_keys.append(f"REGRESSIONS/{process}/{regression.name}")
            regression_key_map.append((process, regression.name, None, None))

    return regression_keys, regression_key_map


def flatten_dict(data: dict, delimiter: str = ":"):
    flat_columns = {}
    shape_metadata = {}

    for key, arr in data.items():
        shape = arr.shape[1:]
        shape_metadata[key] = shape

        if arr.ndim == 1:
            flat_columns[key] = pa.array(arr)
        else:
            flat_arr = arr.reshape(arr.shape[0], -1)
            for i in range(flat_arr.shape[1]):
                suffix = np.unravel_index(i, shape)
                col_key = f"{key}{delimiter}" + delimiter.join(map(str, suffix))
                flat_columns[col_key] = pa.array(flat_arr[:, i])

    table = pa.table(flat_columns)
    return table, shape_metadata


def calculate_extra_variables(data: dict):

    mask = data['INPUTS/Source/MASK']

    num_electron = (data['INPUTS/Source/etag'] * mask).sum(axis=1)
    num_muon = (data['INPUTS/Source/utag'] * mask).sum(axis=1)
    num_bjet = (data['INPUTS/Source/btag'] * mask).sum(axis=1)
    num_jet = (data['INPUTS/Source/qtag'] * mask).sum(axis=1)

    return {
        'EXTRA/num_electron': num_electron,
        'EXTRA/num_muon': num_muon,
        'EXTRA/num_bjet': num_bjet,
        'EXTRA/num_jet': num_jet,
    }

def preprocess(in_dir, store_dir, process_info, unique_id, cfg_dir=None, save: bool = True):
    converted_data = []

    # if hasattr(config, 'event_info'):
    if not global_config.loaded:
        global_config.load_yaml(cfg_dir)

    converted_statistics = PostProcessor(global_config)
    assignment_keys, assignment_key_map = generate_assignment_names(global_config.event_info)
    regression_keys, regression_key_map = generate_regression_names(global_config.event_info)
    unique_process_ids = sorted(set(v['process_id'] for v in global_config.process_info.values()))

    shape_metadata = None

    for process in global_config.process_info:

        matched_data = concatenate_h5_datasets(in_dir)
        matched_data["INFO/VetoDoubleAssign"] = np.ones((matched_data["INPUTS/Source/MASK"].shape[0]), dtype=np.bool)

        if matched_data is None:
            print(f"[WARNING] No matched data for process {process} in dir {in_dir}")
            continue

        print(matched_data)

        converter = EveNetDataConverter(
            raw_data=deepcopy(matched_data),
            event_info=global_config.event_info,
            process=process,
        )

        # Filter the data
        converter.filter_process(process=process, process_info=process_info[process])

        # Load Point Cloud and Mask
        sources = converter.load_sources()
        sources['sources-1-mask'] = sources['sources-1-mask'][:, np.newaxis]
        num_sequential_vectors = np.sum(sources['sources-0-mask'], axis=1)
        num_vectors = num_sequential_vectors  # + np.sum(sources['sources-1-mask'][:, np.newaxis], axis=1) // TODO: tmp fix

        assignments = converter.load_assignments(assignment_keys, assignment_key_map, direct_from_spanet=True)
        regressions = converter.load_regressions(regression_keys, regression_key_map, direct_from_spanet=True)
        classifications = converter.load_classification()

        # invisible = converter.load_invisible(max_num_neutrinos=global_config.get("max_neutrinos", 2))

        # Extra variables
        extra_variables = calculate_extra_variables(converter.raw_data)

        if 'sources-1-data' not in sources:
            sources['sources-1-data'] = np.zeros((sources['sources-0-data'].shape[0], 1), dtype=np.float32)
            sources['sources-1-mask'] = np.ones((sources['sources-0-data'].shape[0], 1), dtype=np.bool)

        process_data = {
            'num_vectors': num_vectors,
            'num_sequential_vectors': num_sequential_vectors,
            'subprocess_id': converter.raw_data['METADATA/PROCESS'],

            'x': sources['sources-0-data'],
            'x_mask': sources['sources-0-mask'],
            'conditions': sources['sources-1-data'],
            'conditions_mask': sources['sources-1-mask'],

            'classification': classifications['classification-EVENT/signal'],

            # 'x_invisible': invisible['invisible-data'],
            # 'x_invisible_mask': invisible['invisible-mask'],
            # 'num_invisible_raw': invisible['invisible-num-raw'],
            # 'num_invisible_valid': invisible['invisible-num-valid'],

            **assignments,
            **regressions,

            **extra_variables,
        }

        flattened_data, meta_data = flatten_dict(process_data)
        # Count the number of unique processes
        # class_counts = np.bincount(process_data['classification'], minlength=len(unique_process_ids))
        # Or simply use the process ID and set the length of the array to the number of unique processes
        class_counts = np.zeros(len(unique_process_ids), dtype=np.int32)
        class_counts[process_info[process]['process_id']] = len(process_data['classification'])

        total_subprocess = global_config.event_info.event_mapping
        subprocess_counts = np.zeros(len(total_subprocess), dtype=np.int32)
        if process in total_subprocess:
            subprocess_counts[list(total_subprocess.keys()).index(process)] = len(process_data['classification'])

        assignment_mask_per_process = {}
        assignment_idx = {key: i for i, key in enumerate(assignment_keys) if f'TARGETS/{process}/' in key}
        for key, i in assignment_idx.items():
            particle = key.split('/')[2]
            assignment_mask_per_process[particle] = assignments['assignments-mask'][:, i]

        if shape_metadata is None:
            shape_metadata = meta_data
        else:
            assert (shape_metadata == meta_data), "Shape metadata mismatch"

        print(f"[INFO] Current table size: {flattened_data.nbytes / 1024 / 1024:.2f} MB")

        converted_data.append(flattened_data)

        # Calculating normalization statistics
        converted_statistics.add(
            x=process_data['x'],
            conditions=process_data['conditions'],
            regression=process_data['regression-data'],
            num_vectors=process_data['num_sequential_vectors'],
            class_counts=class_counts,
            subprocess_counts=subprocess_counts,
            invisible=process_data['x'],  # TODO: tmp fix
        )
        # Add assignment mask
        converted_statistics.add_assignment_mask(process, assignment_mask_per_process)

    if len(converted_data) == 0:
        print(f"[WARNING] No data found for any of the processes in {in_dir}")
        return None

    final_table = pa.concat_tables(converted_data)

    shuffle_indices = np.random.default_rng(31).permutation(final_table.num_rows)
    final_table = final_table.take(pa.array(shuffle_indices))

    if save:
        ### Save to parquet
        pq.write_table(final_table, f"{store_dir}/data_{unique_id}.parquet")

        with open(f"{store_dir}/shape_metadata.json", "w") as f:
            json.dump(shape_metadata, f)

        print(f"[INFO] Final table size: {final_table.nbytes / 1024 / 1024:.2f} MB")
        print(f"[Saving] Saving {final_table.num_rows} rows to {store_dir}/data_{unique_id}.parquet")

        return converted_statistics

    else:
        return converted_statistics, final_table, shape_metadata


def process_single_run(args):
    """
    Process a single run, e.g. Run_2.Dec20/run_yulei_13.
    """
    pretrain_dir, run_folder_name, store_dir, process_info, cfg_dir = args
    run_folder = Path(pretrain_dir) / run_folder_name
    in_tag = f"{Path(pretrain_dir).name}_{run_folder_name}"
    print(f"[INFO] Processing {in_tag}")
    single_statistics = preprocess(run_folder, store_dir, process_info, unique_id=in_tag, cfg_dir=cfg_dir, save=True)

    return single_statistics


def run_parallel(cfg, cfg_dir, num_workers=8):
    tasks = []

    for pretrain_dir in cfg.pretrain_dirs:
        for run_folder in Path(pretrain_dir).iterdir():
            if run_folder.is_dir():
                tasks.append((
                    pretrain_dir,
                    run_folder.name,
                    cfg.store_dir,
                    global_config.process_info,
                    cfg_dir,
                ))

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_run, tasks)
    # Merge statistics
    PostProcessor.merge(
        results,
        regression_names=global_config.event_info.regression_names,
        saved_results_path=cfg.store_dir,
    )


def main(cfg):
    global_config.load_yaml(cfg.preprocess_config)

    def generate_unique_id():
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    os.makedirs(cfg.store_dir, exist_ok=True)

    if cfg.pretrain_dirs is not None:
        print(f"[INFO] Directories to run: {cfg.pretrain_dirs}")

        run_parallel(cfg, cfg.preprocess_config, num_workers=cpu_count() - 1)

    else:
        in_tag = Path(cfg.in_dir).name
        norm_stats = preprocess(
            cfg.in_dir, cfg.store_dir, global_config.process_info, unique_id=in_tag,
            cfg_dir=cfg.preprocess_config, save=True
        )
        if norm_stats is None:
            print(f"[WARNING] No data found for {cfg.in_dir}. Skipping this run.")
            return

        PostProcessor.merge(
            [norm_stats],
            regression_names=global_config.event_info.regression_names,
            saved_results_path=cfg.store_dir,
        )


if __name__ == '__main__':
    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('preprocess_config', help='Path to config file', default='preprocess_pretrain.yaml')
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--store_dir', type=str, default='Storage')
    parser.add_argument(
        '--pretrain_dirs', type=str, nargs='+',
        help='Pretrain directories, accept a list of directories, will force using 2-level directory structure'
    )
    args = parser.parse_args()

    main(args)
