import json
import numpy as np
from pathlib import Path
from functools import partial
from ray.data import Dataset
import ray
from ray.data.dataset import MaterializedDataset

from evenet.control.global_config import global_config
from evenet.dataset.preprocess import process_event_batch
from preprocessing.preprocess import unflatten_dict


def make_process_fn(base_dir: Path):
    """Creates a partial function for batch preprocessing."""
    shape_metadata = json.load(open(base_dir / "shape_metadata.json"))
    return partial(process_event_batch, shape_metadata=shape_metadata, unflatten=unflatten_dict)


def register_dataset(
        parquet_files: list[str],
        process_event_batch_partial,
        platform_info,
        dataset_total: float = 1.0,
        file_shuffling: bool = False,
) -> tuple[Dataset, int]:
    """Registers a Ray dataset, preprocesses it, and returns dataset and event count."""
    ds = ray.data.read_parquet(
        parquet_files,
        override_num_blocks=len(parquet_files) * platform_info.number_of_workers,
        ray_remote_args={
            "num_cpus": 0.5,
        },
        # Disable file-level shuffling for inference
        shuffle="files" if file_shuffling else None,
    )

    total_events = ds.count()
    ds = ds.limit(int(total_events * dataset_total))
    total_events = ds.count()

    ds = ds.map_batches(
        process_event_batch_partial,
        zero_copy_batch=True,
        batch_size=platform_info.batch_size * global_config.platform.prefetch_batches,
    )

    return ds, total_events


def prepare_datasets(
        base_dir: Path,
        process_event_batch_partial,
        platform_info,
        load_all_in_ram: bool = False,
        predict: bool = False,
) -> tuple[Dataset, None, int, None] | tuple[Dataset, Dataset, int, int]:
    """
    Prepares training and validation datasets.

    Returns:
        train_ds, val_ds, train_count, val_count
    """
    parquet_files: list[str] = sorted(map(str, base_dir.glob("*.parquet")))
    train_ratio = global_config.options.Dataset.train_validation_split
    split_index = int(train_ratio * len(parquet_files))
    dataset_limit = global_config.options.Dataset.dataset_limit

    if predict:
        predict_ds, predict_count = register_dataset(
            parquet_files,
            process_event_batch_partial,
            platform_info,
            dataset_limit,
            file_shuffling=False,
        )
        return predict_ds, None, predict_count, None

    if not load_all_in_ram:
        # No global shuffling — preserve file order
        train_files = parquet_files[:split_index]
        val_files = parquet_files[split_index:]

        dataset_kwargs = {
            'process_event_batch_partial': process_event_batch_partial,
            'platform_info': platform_info,
            "dataset_limit": dataset_limit,
            "file_shuffling": True,
        }

        train_ds, train_count = register_dataset(train_files, **dataset_kwargs)
        val_ds, val_count = register_dataset(val_files, **dataset_kwargs)

        return train_ds, val_ds, train_count, val_count

    else:
        ds = ray.data.read_parquet(
            parquet_files,
            override_num_blocks=len(parquet_files) * platform_info.number_of_workers,
            ray_remote_args={"num_cpus": 0.5},
        )

        total_events = ds.count()
        ds = ds.limit(int(total_events * dataset_limit))

        # Shuffle rows (not files)
        ds = ds.random_shuffle(seed=42)

        # Split into train/val by rows
        train_ds, val_ds = ds.split_proportionately([train_ratio])

        train_ds = train_ds.map_batches(
            process_event_batch_partial,
            zero_copy_batch=True,
            batch_size=platform_info.batch_size * global_config.platform.prefetch_batches,
        )
        val_ds = val_ds.map_batches(
            process_event_batch_partial,
            zero_copy_batch=True,
            batch_size=platform_info.batch_size * global_config.platform.prefetch_batches,
        )

        return train_ds, val_ds, train_ds.count(), val_ds.count()
