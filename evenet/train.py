import os
import argparse
import logging
import sys

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
from evenet.dataset.preprocess import process_event_batch
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
    batch_size = config.Dataset.batch_size
    max_epochs = config.Training.epochs

    # Fetch the Dataset shards
    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("validation")

    # Create a dataloader for Ray Datasets
    train_ds_loader = train_ds.iter_torch_batches(batch_size=batch_size)
    val_ds_loader = val_ds.iter_torch_batches(batch_size=batch_size)

    # Model
    model = EveNetEngine()

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        enable_progress_bar=False,
    )

    trainer = prepare_trainer(trainer)

    trainer.fit(model, train_dataloaders=train_ds_loader, val_dataloaders=val_ds_loader)


def main(args):
    # os.environ["RAY_ENABLE_MAC_LARGE_OBJECT_STORE"] = "1"

    config.load_yaml(args.config)

    ray.init(
        num_cpus=10,
        # object_store_memory=10 * 1024 * 1024,
        # local_mode=True,
    )

    ds = ray.data.read_parquet([
        "/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/PreTrain_Parquet/multi_process_0.parquet",
        # "/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/PreTrain_Parquet/multi_process_1.parquet"
    ]).limit(1000)

    processed_ds = ds.map_batches(
        process_event_batch,
        # concurrency=5,
        # batch_format="default"
    )

    run_config = RunConfig(
        name="EveNet Training",
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="max",
        ),
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
            # "validation": validation_dataset
        },
    )

    result = trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EveNet Training Program")
    parser.add_argument("config", help="Path to config file")

    args, _ = parser.parse_known_args()

    main(args)
