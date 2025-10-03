# 📘 EveNet Tutorial: From Zero to First Predictions

New to EveNet? This tutorial walks you through the end-to-end workflow so you can move from a clean checkout to model predictions with confidence. Each step links to a deeper reference guide if you need more detail.

---

## 1. Understand the Project Layout

Before running any commands, skim these key directories:

- `evenet/` – PyTorch Lightning modules, Ray data pipelines, and trainer utilities.
- `preprocessing/` – CLI and helpers for converting raw ntuples into parquet shards and metadata.
- `share/` – Ready-to-edit YAML configurations for preprocessing, training, and prediction.
- `docs/` – Reference documentation that expands on this tutorial.

---

## 2. Set Up Your Environment

1. **Install Python dependencies.** EveNet targets Python 3.10+.
   ```bash
   pip install -r requirements.txt
   ```
2. **Optional:** If you plan to run inside Docker or on NERSC, review the environment helpers under `Docker/` and `NERSC/` for prebuilt container recipes and SLURM launch scripts.
3. **Verify GPU visibility (if available).**
   ```bash
   python -c "import torch; print(torch.cuda.device_count())"
   ```

---

## 3. Prepare Input Data

1. **Start from a preprocessing YAML.** Duplicate `share/preprocess_pretrain.yaml` (or another example) and customize:
   - Campaign directories (`pretrain_dirs` or `in_dir`)
   - Process lists, selections, and padding strategy
   - Output paths (`store_dir`)
2. **Run the preprocessing CLI.**
   ```bash
   python preprocessing/preprocess.py share/preprocess_pretrain.yaml \
     --pretrain_dirs /path/to/run_A /path/to/run_B \
     --store_dir /path/to/output \
     --cpu_max 32
   ```
3. **Inspect outputs.** The command produces `data_*.parquet`, `shape_metadata.json`, and `normalization.pt` artifacts described in the [data preparation guide](data_preparation.md).

> 🔍 Tip: Keep preprocessing configs in version control to document how each dataset was produced.

---

## 4. Configure an Experiment

1. Copy `share/finetune-example.yaml` (for training) and `share/predict-example.yaml` (for inference) into a working directory.
2. Update paths and experiment metadata:
   - `platform.data_parquet_dir` → location of your processed parquet files
   - `Dataset.normalization_file` → normalization statistics file from preprocessing
   - `logger` → project names, WANDB API key, or local log paths
3. Review the [configuration reference](configuration.md) for a description of every YAML section and available overrides.

---

## 5. Train the Model

1. Export your Weights & Biases API key if you plan to log online.
   ```bash
   export WANDB_API_KEY=<your_key>
   ```
2. Launch training with your updated YAML.
   ```bash
   python evenet/train.py path/to/your-train-config.yaml
   ```
3. Monitor progress:
   - Console output provides per-epoch metrics and checkpoint locations.
   - WANDB dashboards (if enabled) visualize loss curves and system stats.
   - Checkpoints and logs are stored under `options.Training.model_checkpoint_save_path`.
4. Consult `docs/train.md` for details on checkpointing, EMA weights, and resume logic.

---

## 6. Generate Predictions

1. Ensure the prediction YAML points to the trained checkpoint via `options.Training.model_checkpoint_load_path`.
2. Launch inference.
   ```bash
   python evenet/predict.py path/to/your-predict-config.yaml
   ```
3. Outputs land in the configured writers (e.g., parquet, numpy archives). See `docs/predict.md` for writer options and schema notes.

---

## 7. Explore and Iterate

- Use the artifacts written in the prediction step for downstream analysis (examples live under `downstreams/`).
- Adjust YAML hyperparameters, architecture templates, or preprocessing selections and repeat the workflow.
- When adding new datasets or modules, contribute documentation updates so the next user can follow your path.

Happy exploring! 🚀
