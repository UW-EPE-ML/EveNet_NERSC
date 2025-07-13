import os
import argparse
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
from ray.train import RunConfig, ScalingConfig, CheckpointConfig

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichModelSummary
from lightning.pytorch.profilers import PyTorchProfiler

from evenet.control.global_config import global_config
from shared import make_process_fn, prepare_datasets, EveNetTrainCallback
from evenet.engine import EveNetEngine


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
        'local_shuffle_buffer_size': batch_size * prefetch_batches,
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
        save_top_k=10,
        mode="min",
        verbose=True,
        dirpath=global_config.options.Training.model_checkpoint_save_path,
        save_last="link",
        auto_insert_metric_name=False,
        filename="epoch={epoch}_train={train/loss:.4f}_val={val/loss:.4f}",
    )
    early_stop_callback = EarlyStopping(
        **cfg.get("early_stopping", {}),
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
        strategy=RayDDPStrategy(find_unused_parameters=True, timeout=180),
        plugins=[RayLightningEnvironment()],
        callbacks=[
            EveNetTrainCallback(),
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
        # profiler=PyTorchProfiler(
        #     dirpath=global_config.options.Training.model_checkpoint_save_path,
        #     filename=f"profiler_{world_rank}",
        # ),
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


def main(args):
    assert (
            "WANDB_API_KEY" in os.environ
    ), 'Please set WANDB_API_KEY="abcde" when running this script.'

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": f"{Path(__file__).resolve().parent.parent}:{os.environ.get('PYTHONPATH', '')}",
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
            # "TORCH_NCCL_BLOCKING_WAIT": "1",
            # "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_TIMEOUT": "180",
            # "NCCL_DEBUG_SUBSYS": "ALL",
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "1000000",
        }
    }

    global_config.load_yaml(args.config)
    global_config.display()

    platform_info = global_config.platform

    ray.init(
        runtime_env=runtime_env,
    )

    base_dir = Path(platform_info.data_parquet_dir)

    process_fn = make_process_fn(base_dir)
    train_ds, valid_ds, total_events, val_count = prepare_datasets(
        base_dir,
        process_fn,
        platform_info,
        args.load_all,
        predict=False,
    )

    run_config = RunConfig(
        name="EveNet-Training",
        storage_path=args.ray_dir,
    )

    # Schedule four workers for DDP training (1 GPU/worker by default)
    scaling_config = ScalingConfig(
        num_workers=platform_info.number_of_workers,
        resources_per_worker=platform_info.resources_per_worker,
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
        "early_stopping": global_config.options.Training.EarlyStopping,
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

    import torch.distributed as dist
    if dist.is_initialized():
      dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EveNet Training Program")
    parser.add_argument("config", help="Path to config file")
    # argument for loading all dataset files into RAM
    parser.add_argument("--load_all", action="store_true", help="Load all dataset files into RAM")
    parser.add_argument("--ray_dir", type=str, default="~/ray_results")

    args, _ = parser.parse_known_args()

    main(args)
