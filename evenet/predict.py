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
from shared import make_process_fn, prepare_datasets


def predict_func(cfg):
    from ray.train import get_dataset_shard, get_context

    predict_ds_ = get_dataset_shard("predict")
    predict_ds_loader = predict_ds_.iter_torch_batches(
        batch_size=cfg['batch_size'],
        prefetch_batches=cfg['prefetch_batches'],
    )

    model = EveNetEngine(
        global_config=global_config,
        world_size=get_context().get_world_size(),
        total_events=cfg['total_events'],
    )

    if global_config.options.Training.model_checkpoint_load_path:
        ckpt_path = global_config.options.Training.model_checkpoint_load_path
        print(f"Loading checkpoint from model_checkpoint_load_path: {ckpt_path}")
    elif global_config.options.Training.pretrain_model_load_path:
        ckpt_path = global_config.options.Training.pretrain_model_load_path
        print(f"Loading checkpoint from pretrain_model_load_path: {ckpt_path}")
    else:
        raise ValueError(
            "Checkpoint path required for prediction, "
            "but neither model_checkpoint_load_path nor pretrain_model_load_path is set."
        )

    accelerator_config = {
        "accelerator": "auto",
        "devices": "auto",
    }
    # if this is macOS, set the accelerator to "cpu"
    if os.uname().sysname == "Darwin":
        accelerator_config["accelerator"] = "cpu"
        accelerator_config["devices"] = 1

    predictor = L.Trainer(
        strategy=RayDDPStrategy(find_unused_parameters=True),
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=True,
        **accelerator_config,
    )

    predictor = prepare_trainer(predictor)

    predictions = predictor.predict(model, dataloaders=predict_ds_loader, ckpt_path=ckpt_path)

    print(
        f"[Rank {get_context().get_world_rank()}] Prediction done: {len(predictions)} batches; Prediction results: {predictions}")


class PredictDataControl(DataConfig):
    def configure(
            self,
            datasets: dict[str, Dataset],
            world_size: int,
            worker_handles: Optional[list[ActorHandle]],
            worker_node_ids: Optional[list[NodeIdStr]],
            **kwargs,
    ) -> list[dict[str, DataIterator]]:
        output = [{} for _ in range(world_size)]

        datasets_to_split = set(self._datasets_to_split)

        locality_hints = (
            worker_node_ids if self._execution_options.locality_with_output else None
        )
        for name, ds in datasets.items():
            execution_options = copy.deepcopy(self._execution_options)

            if execution_options.is_resource_limits_default():
                # If "resource_limits" is not overriden by the user,
                # add training-reserved resources to Data's exclude_resources.
                execution_options.exclude_resources = (
                    execution_options.exclude_resources.add(
                        ExecutionResources(
                            cpu=self._num_train_cpus, gpu=self._num_train_gpus
                        )
                    )
                )

            ds = ds.copy(ds)
            ds.context.execution_options = execution_options

            if name in datasets_to_split:
                for i, split in enumerate(
                        ds.streaming_split(
                            world_size, equal=False, locality_hints=locality_hints
                        )
                ):
                    output[i][name] = split
            else:
                for i in range(world_size):
                    output[i][name] = ds.iterator()

        return output


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

    # predict_ds, _, predict_count, _ = prepare_datasets(
    #     base_dir, process_fn, platform_info, predict=True
    # )

    import pandas as pd

    # Create a dummy dataset with one column
    df = pd.DataFrame({"value": list(range(1, 168))})

    # Convert to Ray Dataset
    predict_ds = ray.data.from_pandas(df)
    predict_count = predict_ds.count()
    print(df)

    trainer = TorchTrainer(
        train_loop_per_worker=predict_func,
        train_loop_config={
            "batch_size": platform_info.batch_size,
            "prefetch_batches": platform_info.prefetch_batches,
            "total_events": predict_count,
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
