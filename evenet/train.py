import os
import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path

import ray
import ray.train
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
from preprocessing.preprocess import unflatten_dict
import json


def train_func(cfg):
    batch_size = cfg['batch_size']
    max_epochs = cfg['epochs']
    prefetch_batches = cfg['prefetch_batches']

    wandb_config = cfg.get("wandb", {})
    wandb_logger = None
    if ray.train.get_context().get_world_rank() == 0:
        wandb_logger = WandbLogger(
            project=wandb_config.get("project", "EveNet"),
            name=wandb_config.get("run_name", None),
            tags=wandb_config.get("tags", []),
            entity=wandb_config.get("entity", None),
        )

    dataset_configs = {
        'batch_size': batch_size,
        'prefetch_batches': prefetch_batches,
    }

    # Fetch the Dataset shards
    train_ds = ray.train.get_dataset_shard("train")
    # val_ds = ray.train.get_dataset_shard("validation")

    train_ds_loader = train_ds.iter_torch_batches(**dataset_configs)
    # val_ds_loader = val_ds.iter_torch_batches(**dataset_configs)

    # Model
    model = EveNetEngine(global_config=global_config)

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="best-{epoch}-{val_loss:.4f}",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # metric to monitor
        patience=5,  # epochs to wait for improvement
        mode="min",  # "min" if lower is better (e.g. for loss)
        verbose=True,  # optional: prints when triggered
        min_delta=0.001,  # minimum change to qualify as improvement
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        # accelerator="cpu",
        # devices=1,
        strategy=RayDDPStrategy(find_unused_parameters=True),
        plugins=[RayLightningEnvironment()],
        callbacks=[
            # RayTrainReportCallback(),
            checkpoint_callback,
            early_stop_callback,
            LearningRateMonitor(),
            DeviceStatsMonitor(),
            RichModelSummary(max_depth=1)
        ],
        enable_progress_bar=True,
        logger=wandb_logger,
    )

    trainer = prepare_trainer(trainer)

    trainer.fit(model, train_dataloaders=train_ds_loader, val_dataloaders=None)


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
    platform_info = global_config.platform

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
        batch_size=platform_info.batch_size * global_config.platform.prefetch_batches,
    )

    run_config = RunConfig(
        name="EveNet Training",
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
        }
    }

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=trainer_config,
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
