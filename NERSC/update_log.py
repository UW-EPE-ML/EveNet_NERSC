import argparse
from pathlib import Path
import os
from tqdm import tqdm
import yaml
import json
import dask.dataframe as dd
import pandas as pd
import wandb

CHECKPOINT_FILE = "wandb_sync_checkpoint.json"


def load_checkpoint(checkpoint_path: Path):
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {"processed_rows": 0}


def save_checkpoint(checkpoint_path: Path, checkpoint: dict):
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def count_rows(filepath: Path):
    with filepath.open("r") as f:
        return sum(1 for _ in f) - 1  # exclude header


def main(args):
    assert "WANDB_API_KEY" in os.environ, 'Please set WANDB_API_KEY="abcde" before running.'

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    if "logger" not in config_data:
        raise KeyError("Missing required config key: 'logger'")
    log_config = config_data["logger"]

    # WandB setup
    wandb_config = log_config["wandb"]
    run = wandb.init(
        project=wandb_config["project"],
        name=wandb_config["run_name"],
        entity=wandb_config.get("entity", None),
        id=wandb_config.get("id", None),
        resume="allow",
    )

    # Path to metric files
    local_config = log_config["local"]
    log_dir = Path(local_config["save_dir"]) / local_config["name"] / str(local_config["version"])
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory {log_dir} does not exist: {log_dir}")

    csv_files = sorted(log_dir.glob("metrics_rank*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV log files found in the specified log directory.")

    # Load global checkpoint
    checkpoint_path = log_dir / CHECKPOINT_FILE
    checkpoint = load_checkpoint(checkpoint_path)
    processed_rows = checkpoint.get("processed_rows", 0)

    # Find the minimum row count (truncate to avoid partial step averaging)
    min_rows = min(count_rows(f) for f in csv_files)
    if processed_rows >= min_rows:
        print(f"No new complete rows to process (processed_rows={processed_rows}, min_rows={min_rows})")
        return

    print(f"Processing rows from {processed_rows} to {min_rows - 1}")

    ddfs = []
    for f in csv_files:
        if processed_rows > 0:
            skip = range(1, processed_rows + 1)
        else:
            skip = None  # Don't pass skiprows if nothing to skip

        ddf = dd.read_csv(
            f,
            skiprows=skip,
            assume_missing=True,
            blocksize=None,
        )
        ddfs.append(ddf)

    combined_ddf = dd.concat(ddfs, axis=0, interleave_partitions=True)

    # Group and average by keys
    group_keys = ["batch", "epoch", "step", "training"]
    averaged_ddf = combined_ddf.groupby(group_keys).mean(numeric_only=True)
    averaged_df = averaged_ddf.compute().reset_index()

    # Step 1: Separate train/val
    train_df = averaged_df[averaged_df["training"] == 1].copy()
    val_df = averaged_df[averaged_df["training"] == 0].copy()

    # Step 2: Drop val columns from train_df
    val_cols = [col for col in train_df.columns if col.startswith("val/")]
    train_df.drop(columns=val_cols, inplace=True)

    # Step 3: Map each epoch to the max training step
    epoch_to_step = (
        train_df.groupby("epoch")["step"]
        .max()
        .dropna()
        .astype(int)
        .to_dict()
    )

    # Step 4: Assign matching step to validation rows
    val_df["step"] = val_df["epoch"].map(epoch_to_step)

    # Step 5: Group val_df by step, keeping epoch, and averaging val metrics
    val_metric_cols = [col for col in val_df.columns if col.startswith("val/")]
    val_grouped = (
        val_df.groupby("step").agg(
            {
                "epoch": "first",
                **{col: "mean" for col in val_metric_cols}
            }
        ).reset_index()
    )

    # Step 6: Merge back train and val into a single DataFrame
    merged_df = pd.merge(train_df, val_grouped, on=["step", "epoch"], how="left")
    merged_df.drop(columns=['training', 'batch'], inplace=True)
    # set "epoch" and "step" as int
    merged_df["epoch"] = merged_df["epoch"].astype(int)
    merged_df["step"] = merged_df["step"].astype(int)
    merged_df.sort_values(by=["step", "epoch"], inplace=True)

    run.define_metric("*", step_metric='trainer/global_step')

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Logging to WandB"):
        row["trainer/global_step"] = row["step"]
        log_data = {
            k: v
            for k, v in row.items()
            # if (k.startswith("train/") or k.startswith("val/")) and ~np.isnan(v)
            if not pd.isna(v)
        }
        if log_data:
            # wandb.log(log_data, step=step)
            run.log(log_data)

    # Save the averaged DataFrame to the log directory
    averaged_csv_path = log_dir / "averaged_metrics.csv"
    merged_df.to_csv(averaged_csv_path, index=False)

    # Update checkpoint
    checkpoint["processed_rows"] = min_rows
    save_checkpoint(checkpoint_path, checkpoint)
    wandb.finish()
    print(f"Logged {len(averaged_df)} averaged rows to WandB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EveNet WandB Log Sync")
    parser.add_argument("config", help="Path to YAML config file")
    args, _ = parser.parse_known_args()
    main(args)
