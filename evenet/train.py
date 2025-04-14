import os
import argparse
import sys
from copy import deepcopy
from functools import partial
from io import StringIO
from pathlib import Path

import wandb
from rich.console import Console

import numpy as np
import ray
import ray.train
from ray.data import Dataset
from ray.train.lightning import (
    prepare_trainer,
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, \
    RichModelSummary

from evenet.control.global_config import global_config
from evenet.dataset.preprocess import process_event_batch
from evenet.engine import EveNetEngine
from evenet.network.callbacks.ema import EMACallback
from preprocessing.preprocess import unflatten_dict
import json

from rich.console import Console
import lightning.pytorch.utilities.model_summary


def train_func(cfg):
    batch_size = cfg['batch_size']
    max_epochs = cfg['epochs']
    prefetch_batches = cfg['prefetch_batches']
    total_events = cfg['total_events']
    world_rank = ray.train.get_context().get_world_rank()

    wandb_config = cfg.get("wandb", {})
    wandb_logger = WandbLogger(
        project=wandb_config.get("project", "EveNet"),
        name=wandb_config.get("run_name", None),
        tags=wandb_config.get("tags", []),
        entity=wandb_config.get("entity", None),
        config=global_config.to_logger()
    )
    # wandb_logger.experiment.config.update()

    dataset_configs = {
        'batch_size': batch_size,
        'prefetch_batches': prefetch_batches,
    }

    # Fetch the Dataset shards
    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("validation")

    train_ds_loader = train_ds.iter_torch_batches(**dataset_configs)
    val_ds_loader = val_ds.iter_torch_batches(**dataset_configs)

    # Model
    model = EveNetEngine(
        global_config=global_config,
        world_size=ray.train.get_context().get_world_size(),
        total_events=total_events,
    )

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        verbose=True,
        dirpath=global_config.options.Training.model_checkpoint_save_path,
        filename="best_{epoch}",
    )
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=10,  # epochs to wait for improvement
        mode="min",  # "min" if lower is better (e.g. for loss)
        verbose=True,  # optional: prints when triggered
        min_delta=0.001,  # minimum change to qualify as improvement
    )

    accelerator_config = {
        "accelerator": "auto",
        "devices": "auto",
    }
    # if this is macOS, set the accelerator to "cpu"
    if os.uname().sysname == "Darwin":
        accelerator_config["accelerator"] = "cpu"
        accelerator_config["devices"] = 1

    trainer = L.Trainer(
        max_epochs=max_epochs,
        strategy=RayDDPStrategy(find_unused_parameters=True),
        plugins=[RayLightningEnvironment()],
        callbacks=[
            RayTrainReportCallback(),
            checkpoint_callback,
            early_stop_callback,
            LearningRateMonitor(),
            RichModelSummary(max_depth=3),
        ],
        enable_progress_bar=True,
        logger=wandb_logger,
        # val_check_interval=10,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        **accelerator_config,
    )

    trainer = prepare_trainer(trainer)

    ckpt_path = None
    if global_config.options.Training.model_checkpoint_load_path is not None:
        ckpt_path = global_config.options.Training.model_checkpoint_load_path
        print(f"Loading checkpoint from {ckpt_path}")

    trainer.fit(
        model,
        train_dataloaders=train_ds_loader,
        val_dataloaders=val_ds_loader,
        ckpt_path=ckpt_path,
    )


def register_dataset(
        parquet_files: list[str],
        process_event_batch_partial,
        platform_info,
        dataset_total: float = 1.0
) -> tuple[Dataset, int]:
    # Create Ray datasets
    ds = ray.data.read_parquet(
        parquet_files,
        override_num_blocks=len(parquet_files) * platform_info.number_of_workers,
        ray_remote_args={
            "num_cpus": 0.5,
        },
        shuffle="files",
    )

    total_events = ds.count()

    ds = ds.limit(int(total_events * dataset_total))
    total_events = ds.count()

    ds = ds.map_batches(
        process_event_batch_partial,
        # batch_format="pyarrow",
        zero_copy_batch=True,
        batch_size=platform_info.batch_size * global_config.platform.prefetch_batches,
    )

    return ds, total_events


def main(args):
    assert (
            "WANDB_API_KEY" in os.environ
    ), 'Please set WANDB_API_KEY="abcde" when running this script.'

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": f"{Path(__file__).resolve().parent.parent}:{os.environ.get('PYTHONPATH', '')}",
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"]
        }
    }

    global_config.load_yaml(args.config)
    global_config.display()

    platform_info = global_config.platform

    ray.init(
        runtime_env=runtime_env,
    )

    base_dir = Path(platform_info.data_parquet_dir)

    shape_metadata = json.load(open(base_dir / "shape_metadata.json"))

    process_event_batch_partial = partial(process_event_batch, shape_metadata=shape_metadata, unflatten=unflatten_dict)

    # Collect all .parquet file paths
    parquet_files = sorted(base_dir.glob("*.parquet"))  # sort for reproducibility
    parquet_files = list(map(str, parquet_files))  # convert to str if needed

    # Shuffle the file list
    np.random.seed(42)
    np.random.shuffle(parquet_files)

    # Split the file list
    split_index = int(global_config.options.Dataset.train_validation_split * len(parquet_files))
    train_files = parquet_files[:split_index]
    val_files = parquet_files[split_index:]

    # Create Ray datasets
    dataset_limit = global_config.options.Dataset.dataset_limit
    train_ds, total_events = register_dataset(train_files, process_event_batch_partial, platform_info, dataset_limit)
    valid_ds, _ = register_dataset(val_files, process_event_batch_partial, platform_info, dataset_limit)

    run_config = RunConfig(
        name="EveNet-Training",
    )

    # Schedule four workers for DDP training (1 GPU/worker by default)
    scaling_config = ScalingConfig(
        num_workers=platform_info.number_of_workers,
        resources_per_worker=platform_info.resources_per_worker,
        # use_gpu=False,
        use_gpu=platform_info.get("use_gpu", True),
    )

    trainer_config = {
        "batch_size": platform_info.batch_size,
        "epochs": global_config.options.Training.epochs,
        "prefetch_batches": platform_info.prefetch_batches,
        'wandb': {
            **global_config.wandb,
        },
        "total_events": total_events,
    }

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=trainer_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={
            "train": train_ds,
            "validation": valid_ds,
        },
    )

    result = trainer.fit()

    print('finished!')
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EveNet Training Program")
    parser.add_argument("config", help="Path to config file")

    args, _ = parser.parse_known_args()

    main(args)
