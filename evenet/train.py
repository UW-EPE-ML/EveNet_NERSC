import os
import argparse
import logging
import sys
from pathlib import Path

import ray
from ray import air, tune
from ray.air import session
from ray.train.torch import TorchCheckpoint
from ray.tune.schedulers import AsyncHyperBandScheduler
import ray.train
from ray.train.lightning import (
    prepare_trainer,
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig, DataConfig

import wandb

import lightning as L

from evenet.control.config import config, DotDict, Config
from evenet.dataset.preprocess import process_event
from evenet.engine import EveNetEngine


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
    # Unpack the input configs passed from `TorchTrainer(train_loop_config)`
    # batch_size = config.Dataset.batch_size
    # max_epochs = config.Training.epochs
    batch_size = 4096
    max_epochs = 5

    # Fetch the Dataset shards
    train_ds = ray.train.get_dataset_shard("train")
    # val_ds = ray.train.get_dataset_shard("validation")

    # Create a dataloader for Ray Datasets
    dataset_configs = {
        'batch_size': batch_size,
        # 'collate_fn': process_event,
        'prefetch_batches': 10,
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

    # config.load_yaml(args.config)

    ray.init(
        # num_cpus=10,
        # object_store_memory=10 * 1024 * 1024,
        # local_mode=True,
        runtime_env=runtime_env,
    )

    base_dir = Path("/global/cfs/cdirs/m2616/tihsu/PreTrain_Parquet")

    parquet_files = [
        str(base_dir / file) for file in base_dir.glob("*.parquet")
    ]

    ds = ray.data.read_parquet(
        parquet_files,
        override_num_blocks=len(parquet_files),
        ray_remote_args={
            "num_cpus": 0.25,
        }
    )

    ds = ds.map_batches(process_event)

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
        num_workers=10,
        resources_per_worker={
            "CPU": 32,
            "GPU": 1,
        },
        use_gpu=True
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        # train_loop_config=config,
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
