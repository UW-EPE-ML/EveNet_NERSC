import copy
import os
from pathlib import Path
from typing import Optional

import ray
from ray.actor import ActorHandle
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer

from ray.train import DataConfig, ScalingConfig
from ray.data import Dataset, DataIterator, NodeIdStr, ExecutionResources

import lightning as L
from evenet.control.global_config import global_config
from evenet.engine import EveNetEngine
from evenet.network.callbacks.predict_writer import PredWriter
from shared import make_process_fn, prepare_datasets


def predict_func(cfg):
    from ray.train import get_dataset_shard, get_context

    predict_ds_ = get_dataset_shard("predict")
    predict_ds_loader = predict_ds_.iter_torch_batches(
        batch_size=cfg['batch_size'],
        prefetch_batches=cfg['prefetch_batches'],
    )

    if global_config.options.Training.model_checkpoint_load_path:
        ckpt_path = global_config.options.Training.model_checkpoint_load_path
        print(f"Loading checkpoint from model_checkpoint_load_path: {ckpt_path}")
    elif global_config.options.Training.pretrain_model_load_path:
        ckpt_path = None
    else:
        raise ValueError(
            "Checkpoint path required for prediction, "
            "but neither model_checkpoint_load_path nor pretrain_model_load_path is set."
        )

    model = EveNetEngine(
        global_config=global_config,
        world_size=get_context().get_world_size(),
        total_events=cfg['total_events'],
    )

    accelerator_config = {
        "accelerator": "auto",
        "devices": "auto",
    }
    # if this is macOS, set the accelerator to "cpu"
    if os.uname().sysname == "Darwin":
        accelerator_config["accelerator"] = "cpu"
        accelerator_config["devices"] = 1

    callbacks = []
    predict_write_config = global_config.options.get("prediction", None)
    if predict_write_config:
        pred_writer = PredWriter(
            output_dir=Path(cfg['current_dir']) / predict_write_config["output_dir"],
            filename=predict_write_config["filename"],
        )
        callbacks.append(pred_writer)
    else:
        print("No prediction writer config found, skipping prediction writing.")

    predictor = L.Trainer(
        strategy=RayDDPStrategy(find_unused_parameters=True),
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=True,
        callbacks=callbacks,
        **accelerator_config,
    )

    predictor = prepare_trainer(predictor)

    predictions = predictor.predict(model, dataloaders=predict_ds_loader, ckpt_path=ckpt_path)

    print(f"[Rank {get_context().get_world_rank()}] Prediction done: {len(predictions)} batches;")


class PredictDataControl(DataConfig):
    def configure(
            self,
            datasets: dict[str, Dataset],
            world_size: int,
            worker_handles: Optional[list[ActorHandle]],
            worker_node_ids: Optional[list[NodeIdStr]],
            **kwargs,
    ) -> list[dict[str, DataIterator]]:
        # Configure Ray Data for ingest.
        ctx = ray.data.DataContext.get_current()
        ctx.execution_options = DataConfig.default_ingest_options()

        name, ds = next(iter(datasets.items()))

        # Repartition to ensure enough blocks
        ds = ds.repartition(world_size)

        # Perform streaming split
        iterator_shards = ds.streaming_split(
            world_size, equal=False, locality_hints=worker_node_ids
        )

        # Return shards, each wrapped with the dataset key
        return [{name: it} for it in iterator_shards]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="EveNet Prediction Program")
    parser.add_argument("config", help="Path to config YAML")

    args = parser.parse_args()

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": f"{Path(__file__).resolve().parent}:{os.environ.get('PYTHONPATH', '')}",
        }
    }

    global_config.load_yaml(args.config)
    global_config.display()

    ray.init(runtime_env=runtime_env)
    platform_info = global_config.platform
    base_dir = Path(platform_info.data_parquet_dir)

    process_fn = make_process_fn(base_dir)

    predict_ds, _, predict_count, _ = prepare_datasets(
        base_dir, process_fn, platform_info, predict=True
    )

    trainer = TorchTrainer(
        train_loop_per_worker=predict_func,
        train_loop_config={
            "batch_size": platform_info.batch_size,
            "prefetch_batches": platform_info.prefetch_batches,
            "total_events": predict_count,
            "current_dir": os.getcwd(),
        },
        scaling_config=ScalingConfig(
            num_workers=platform_info.number_of_workers,
            resources_per_worker=platform_info.resources_per_worker,
            use_gpu=platform_info.get("use_gpu", True),
        ),
        run_config=RunConfig(name="EveNet-Predict"),
        datasets={"predict": predict_ds},
        dataset_config=PredictDataControl()
    )

    result = trainer.fit()
    print("Prediction finished.")

    import time

    print("Sleeping to allow Ray shutdown...")
    time.sleep(2)
