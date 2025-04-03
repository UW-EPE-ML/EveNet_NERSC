import os
import argparse
import logging
import sys
from functools import partial
from pathlib import Path

import ray
from ray.train.torch import TorchCheckpoint
import ray.train
from ray.train.lightning import (
    prepare_trainer,
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig, DataConfig
import torch

import wandb

import lightning as L

from evenet.control.config import config, DotDict, Config
from evenet.dataset.preprocess import process_event_batch
from evenet.engine import EveNetEngine
from preprocessing.preprocess import unflatten_dict
import json


def setup_logger(log_file: str = "output.log", level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)

    for handler in [file_handler, stream_handler]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def train_func():
    batch_size = config.platform.batch_size
    max_epochs = config.options.Training.epochs

    # Fetch the Dataset shards
    train_ds = ray.train.get_dataset_shard("train")
    # val_ds = ray.train.get_dataset_shard("validation")

    # Create a dataloader for Ray Datasets
    dataset_configs = {
        'batch_size': batch_size,
        # 'collate_fn': process_event,
        'prefetch_batches': config.platform.prefetch_batches,
    }

    shard_stats = train_ds.stats()
    print(f"[Rank {ray.train.get_context().get_world_rank()}] Dataset shard stats: {shard_stats}")

    train_ds_loader = train_ds.iter_torch_batches(**dataset_configs)
    # val_ds_loader = val_ds.iter_torch_batches(**dataset_configs)

    # Model
    model = EveNetEngine()

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        enable_progress_bar=True,
    )

    trainer = prepare_trainer(trainer)

    trainer.fit(model, train_dataloaders=train_ds_loader, val_dataloaders=None)


def main(args):
    runtime_env = {
        "env_vars": {
            "PYTHONPATH": f"{Path(__file__).resolve().parent.parent}:{os.environ.get('PYTHONPATH', '')}"
        }
    }

    config.load_yaml(args.config)
    platform_info = config.platform

    ray.init(
        runtime_env=runtime_env,
    )

    base_dir = Path(platform_info.data_parquet_dir)

    parquet_files = [
        str(base_dir / file) for file in base_dir.glob("*.parquet")
    ]

    shape_metadata = json.load(open(base_dir / "shape_metadata.json"))

    ds = ray.data.read_parquet(
        parquet_files,
        override_num_blocks=len(parquet_files) * platform_info.number_of_workers,
        ray_remote_args={
            "num_cpus": 0.5,
        }
    )

    process_event_batch_partial = partial(process_event_batch, shape_metadata=shape_metadata, unflatten=unflatten_dict)

    ds = ds.map_batches(
        process_event_batch_partial,
        # batch_format="pyarrow",
        zero_copy_batch=True,
        batch_size=platform_info.batch_size * config.platform.prefetch_batches,
    )

    run_config = RunConfig(
        name="EveNet Training",
        # checkpoint_config=CheckpointConfig(
        #     num_to_keep=2,
        #     checkpoint_score_attribute="val_loss",
        #     checkpoint_score_order="max",
        # ),
    )

    # Schedule four workers for DDP training (1 GPU/worker by default)
    scaling_config = ScalingConfig(
        num_workers=platform_info.number_of_workers,
        resources_per_worker=platform_info.resources_per_worker,
        use_gpu=True
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={
            "train": ds,
            # "validation": ds,
        },
    )

    result = trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EveNet Training Program")
    parser.add_argument("config", help="Path to config file")

    args, _ = parser.parse_known_args()

    main(args)
