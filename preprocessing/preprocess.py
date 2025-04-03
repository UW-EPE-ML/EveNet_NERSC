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

from evenet.control.config import config
from evenet.control.event_info import EventInfo
from preprocessing.monitor_gen_matching import monitor_gen_matching
# from preprocessing.process_info import Feynman_diagram
from preprocessing.evenet_data_converter import EveNetDataConverter


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

    common_process = set(event_info.product_particles) & set(event_info.regressions)
    # Collect all possible regression keys
    for process in sorted(common_process):
        for particle, regression in event_info.regressions[process].items():
            products = list(regression.items()) if isinstance(regression, dict) else [(None, regression)]
            for product, targets in products:
                for target in targets:
                    if product is not None:
                        key = f"{process}/{particle}/{product}/{target.name}"
                    else:
                        key = f"{process}/{particle}/{target.name}"
                    regression_keys.append(key)
                    regression_key_map.append((process, particle, product, target.name))

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


def unflatten_dict(table: dict[str, np.ndarray], shape_metadata: dict, delimiter: str = ":"):
    reconstructed = {}
    grouped = {}
    for col in table:
        base = col.split(delimiter)[0]
        grouped.setdefault(base, []).append(col)
    for base, columns in grouped.items():
        if base not in shape_metadata:
            reconstructed[base] = table[columns[0]]
        else:
            shape = tuple(shape_metadata[base])
            sorted_columns = sorted(columns, key=lambda x: tuple(map(int, x.split(delimiter)[1:])))
            flat = np.stack([table[col] for col in sorted_columns], axis=1)
            full_shape = (flat.shape[0],) + shape
            reconstructed[base] = flat.reshape(full_shape)
    # Print shapes
    # for k, v in reconstructed.items():
    #     print(f"{k}: {v.shape}")

    return reconstructed


def masked_stats(arr):
    mask = arr != 0
    values = np.where(mask, arr, 0)

    count = mask.sum(axis=0)
    sum_ = values.sum(axis=0)
    sumsq = (values ** 2).sum(axis=0)

    return {"sum": sum_, "sumsq": sumsq, "count": count}


def merge_stats(stats_list):
    def merge_two(a, b):
        return {
            "sum": a["sum"] + b["sum"],
            "sumsq": a["sumsq"] + b["sumsq"],
            "count": a["count"] + b["count"]
        }

    def compute_mean_std(agg):
        count = agg["count"]
        sum_ = agg["sum"]
        sumsq = agg["sumsq"]

        # Avoid divide-by-zero by substituting 1 where count is 0 (doesn't matter because we mask later)
        safe_count = np.where(count == 0, 1, count)

        mean = sum_ / safe_count
        std = np.sqrt(sumsq / safe_count - mean ** 2)

        # Zero out mean/std where count == 0
        mean = np.where(count == 0, 0.0, mean)
        std = np.where(count == 0, 0.0, std)

        return {'mean': mean, 'std': std}

    # Accumulate across all files
    total = {
        "x": None,
        "conditions": None,
        "regression-data": None,
    }

    for s in stats_list:
        for key in ["x", "conditions", "regression-data"]:
            if total[key] is None:
                total[key] = s[key]
            else:
                total[key] = merge_two(total[key], s[key])

    # Final result
    result = {
        "x": compute_mean_std(total["x"]),
        "conditions": compute_mean_std(total["conditions"]),
        "regression-data": compute_mean_std(total["regression-data"]),
    }
    return result


def preprocess(in_dir, store_dir, process_info, unique_id, global_config=None):
    converted_data = []
    converted_statistics = []
    # if hasattr(config, 'event_info'):
    config.load_yaml(global_config)

    assignment_keys, assignment_key_map = generate_assignment_names(config.event_info)
    regression_keys, regression_key_map = generate_regression_names(config.event_info)

    shape_metadata = None

    for process in config.process_info:
        # for process in ["QCD", "TT2L", "TT1L"]:
        # print("Processing ", process)
        matched_data = monitor_gen_matching(
            in_dir=in_dir,
            process=process,
            feynman_diagram_process=config.process_info[process],
            monitor_plots=False,
        )

        if matched_data is None:
            continue

        converter = EveNetDataConverter(
            raw_data=deepcopy(matched_data),
            event_info=config.event_info,
            process=process,
        )

        # Filter the data
        converter.filter_process(process_info[process])

        # Load Point Cloud and Mask
        sources = converter.load_sources()
        num_sequential_vectors = np.sum(sources['sources-0-mask'], axis=1)
        num_vectors = num_sequential_vectors + np.sum(sources['sources-1-mask'][:, np.newaxis], axis=1)

        assignments = converter.load_assignments(assignment_keys, assignment_key_map)
        regressions = converter.load_regressions(regression_keys, regression_key_map)
        classifications = converter.load_classification()

        process_data = {
            'num_vectors': num_vectors,
            'num_sequential_vectors': num_sequential_vectors,

            'x': sources['sources-0-data'],
            'x_mask': sources['sources-0-mask'],
            'conditions': sources['sources-1-data'],
            'conditions_mask': sources['sources-1-mask'][:, np.newaxis],

            'classification': classifications['classification-EVENT/signal'],

            **assignments,
            **regressions,
        }

        flattened_data, meta_data = flatten_dict(process_data)

        if shape_metadata is None:
            shape_metadata = meta_data
        else:
            assert (shape_metadata == meta_data), "Shape metadata mismatch"

        print(f"[INFO] Current table size: {flattened_data.nbytes / 1024 / 1024:.2f} MB")

        converted_data.append(flattened_data)

        # Calculating normalization statistics
        x_stats = masked_stats(process_data['x'].reshape(-1, process_data['x'].shape[-1]))
        cond_stats = masked_stats(process_data['conditions'])
        regression_stats = masked_stats(process_data['regression-data'])

        converted_statistics.append({"x": x_stats, "conditions": cond_stats, "regression-data": regression_stats})

    final_table = pa.concat_tables(converted_data)

    shuffle_indices = np.random.default_rng(42).permutation(final_table.num_rows)
    final_table = final_table.take(pa.array(shuffle_indices))

    ### Save to parquet
    pq.write_table(final_table, f"{store_dir}/data_{unique_id}.parquet")

    with open(f"{store_dir}/shape_metadata.json", "w") as f:
        json.dump(shape_metadata, f)

    print(f"[INFO] Final table size: {final_table.nbytes / 1024 / 1024:.2f} MB")
    print(f"[Saving] Saving {shuffle_indices.size} rows to {store_dir}/data_{unique_id}.parquet")

    return converted_statistics


def process_single_run(args):
    pretrain_dir, run_folder_name, store_dir, process_info, global_config = args
    run_folder = Path(pretrain_dir) / run_folder_name
    in_tag = f"{Path(pretrain_dir).name}_{run_folder_name}"
    print(f"[INFO] Processing {in_tag}")
    single_statistics = preprocess(run_folder, store_dir, process_info, unique_id=in_tag, global_config=global_config)

    return single_statistics


def run_parallel(cfg, global_config, num_workers=8):
    tasks = []

    for pretrain_dir in cfg.pretrain_dirs:
        for run_folder in Path(pretrain_dir).iterdir():
            if run_folder.is_dir():
                tasks.append((
                    pretrain_dir,
                    run_folder.name,
                    cfg.store_dir,
                    config.process_info,
                    global_config,
                ))

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_run, tasks)

    print(results)

    # Merge statistics
    merged_statistics = merge_stats([item for sublist in results for item in sublist])
    with open(f"{cfg.store_dir}/normalization.json", "w") as f:
        json.dump(merged_statistics, f)


def main(cfg):
    config.load_yaml(cfg.preprocess_config)

    def generate_unique_id():
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    os.makedirs(cfg.store_dir, exist_ok=True)

    if cfg.pretrain_dirs is not None:
        print(f"[INFO] Directories to run: {cfg.pretrain_dirs}")

        run_parallel(cfg, cfg.preprocess_config, num_workers=60)
        # run_parallel(cfg, cfg.preprocess_config, num_workers=10)

    else:
        in_tag = Path(cfg.in_dir).name
        preprocess(cfg.in_dir, cfg.store_dir, config.process_info, unique_id=in_tag)


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
