import json
import numpy as np
from pathlib import Path
from functools import partial
from ray.data import Dataset
import ray
from ray.data.dataset import MaterializedDataset
from rich.table import Table
import rich

import os
from tempfile import TemporaryDirectory

from lightning.pytorch.callbacks import Callback

import ray
import ray.train
from ray.train import Checkpoint

from evenet.control.global_config import global_config
from evenet.dataset.preprocess import process_event_batch
from evenet.utilities.tool import unflatten_dict
import logging

def make_process_fn(base_dir: Path):
    """Creates a partial function for batch preprocessing."""
    shape_metadata = json.load(open(base_dir / "shape_metadata.json"))

    drop_column_prefix = ["EXTRA/"]
    if 'extra_save' in global_config.options.get('prediction', {}):
        drop_column_prefix = None

    component_map = {
#        "Classification": "classification",
        "Regression": "regression-",
        "Assignment": "assignments-",
        # "TruthGeneration": "event_generation-",
        # "ReconGeneration": "event_generation-",
        "Segmentation": "segmentation-",
    }

    for name, prefix in component_map.items():
        component = getattr(global_config.options.Training.Components, name)
        if drop_column_prefix and not getattr(component, "include", False):
            drop_column_prefix.append(prefix)

    logging.warning(f"Dropping columns: {drop_column_prefix}")

    return partial(
        process_event_batch,
        shape_metadata=shape_metadata,
        unflatten=unflatten_dict,
        drop_column_prefix=drop_column_prefix,
    )


def register_dataset(
        parquet_files: list[str],
        process_event_batch_partial,
        platform_info,
        dataset_limit: float = 1.0,
        file_shuffling: bool = False,
) -> tuple[Dataset, int]:
    """Registers a Ray dataset, preprocesses it, and returns dataset and event count."""
    ds = ray.data.read_parquet(
        parquet_files,
        override_num_blocks=len(parquet_files) * min(platform_info.number_of_workers, 8),
        ray_remote_args={
            "num_cpus": 0.5,
        },
        # Disable file-level shuffling for inference
        shuffle="files" if file_shuffling else None,
    )

    if dataset_limit < 1.0:
        total_events = ds.count()
        ds = ds.limit(int(total_events * dataset_limit))
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
    val_split = global_config.options.Dataset.val_split
    val_start_index = int(len(parquet_files) * val_split[0])
    val_end_index = int(len(parquet_files) * val_split[1])
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
        val_files = parquet_files[val_start_index:val_end_index]
        train_files = parquet_files[:val_start_index] + parquet_files[val_end_index:]

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
        total_events = ds.count()
        splits = ds.split_at_indices([int(val_split[0] * total_events), int(val_split[1] * total_events)])
        val_ds = splits[1]
        train_ds = splits[0].union(splits[2])

        # Shuffle rows (not files)
        train_ds = train_ds.random_shuffle(seed=42)
        val_ds = val_ds.random_shuffle(seed=42)

        # Create a nice table
        train_events = train_ds.count()
        val_events = val_ds.count()
        split_idx_0, split_idx_1 = int(val_split[0] * total_events), int(val_split[1] * total_events)
        table = Table(title="Dataset Split Summary")

        table.add_column("Split", style="cyan", no_wrap=True)
        table.add_column("Events", style="magenta")
        table.add_column("Fraction", style="green")
        table.add_column("Index Range", style="yellow")
        table.add_row(
            "Validation",
            f"{val_events:,}",
            f"{val_events / total_events:.2%}",
            f" {split_idx_0} - {split_idx_1}",
        )
        table.add_row(
            "Train",
            f"{train_events:,}",
            f"{train_events / total_events:.2%}",
            (f" 0-{split_idx_0}" if not (0 == split_idx_0) else "") +
            (f" {split_idx_1}-{total_events}" if not (split_idx_1 == total_events) else ""),
        )

        # Print it
        rich.print(table)

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



class EveNetTrainCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Fetch metrics from `self.log(..)` in the LightningModule
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items()}

        # Add customized metrics
        metrics["epoch"] = trainer.current_epoch

        # Report to train session
        ray.train.report(metrics=metrics, checkpoint=None)