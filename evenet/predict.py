import os
from pathlib import Path
import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer

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

    print(f"[Rank {get_context().get_world_rank()}] Prediction done: {len(predictions)} batches; Prediction results: {predictions}")


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
    )

    result = trainer.fit()
    print("Prediction finished.")
