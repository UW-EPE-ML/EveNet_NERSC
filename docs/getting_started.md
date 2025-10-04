# 📘 EveNet Tutorial: From Zero to First Predictions

New to EveNet? This tutorial walks you through the end-to-end workflow so you can move from a clean checkout to model predictions with confidence. Each step links to a deeper reference guide if you need more detail.

---

## 1. Choose Your Setup Path

### Option A — Quick Start (Docker + PyPI)

Ideal when you want the official binaries and ready-made CLIs without touching the source code.

1. Pull the runtime image that bundles CUDA, PyTorch, Ray, and common utilities.
   ```bash
   docker pull docker.io/avencast1994/evenet:1.5
   docker run --gpus all -it \
     -v /path/to/your/data:/workspace/data \
     docker.io/avencast1994/evenet:1.5
   ```
2. Inside the container (or any GPU-ready Python 3.12+ environment), install EveNet from PyPI.
   ```bash
   pip install evenet
   ```
3. Invoke the packaged CLIs with your configuration files.
   ```bash
   evenet-train share/finetune-example.yaml --ray_dir ~/ray_results
   evenet-predict share/predict-example.yaml
   ```

This path is “plug and play”—you only manage YAML configs and data paths.

### Option B — Advanced (Source Checkout)

Choose this when you want to edit the Lightning modules, extend datasets, or customize the CLI behavior.

1. Clone the repository (or mount it inside the Docker image from Option A).
   ```bash
   git clone https://github.com/UW-ePE-ML/EveNet_Public.git
   cd EveNet_Public
   ```
2. Reuse the provided Docker image **or** create your own environment on Python 3.12+.
   - Docker: bind mount your checkout and data into the container so code changes persist.
   - Native install: `pip install -r requirements.txt` (plus any CUDA/PyTorch builds required by your system).
3. Run the CLIs straight from source when iterating rapidly.
   ```bash
   python -m evenet.train share/finetune-example.yaml --ray_dir ~/ray_results
   python -m evenet.predict share/predict-example.yaml

   # or call the scripts directly
   python evenet/train.py share/finetune-example.yaml --ray_dir ~/ray_results
   python evenet/predict.py share/predict-example.yaml
   ```

Both options are interoperable—you can install the PyPI package for quick tests and then switch to the cloned source for deeper development.

---

## 2. Understand the Project Layout (Advanced Users)

Before running any commands, skim these key directories:

- `evenet/` – PyTorch Lightning modules, Ray data pipelines, and trainer utilities.
- `share/` – Ready-to-edit YAML configurations for fine-tuning and prediction.
- `docs/` – Reference documentation that expands on this tutorial. Start with [Model Architecture Tour](model_architecture.md) to see how point-cloud and global features flow through EveNet.
- `downstreams/` – Example downstream analysis scripts built on top of EveNet outputs.

---

## 3. Verify Your Environment

1. **Review cluster helpers as needed.** The `Docker/` and `NERSC/` directories include recipes and SLURM launch scripts tailored for HPC environments.
2. **Confirm GPU visibility (if available).**
   ```bash
   python -c "import torch; print(torch.cuda.device_count())"
   ```

---

## 4. Download Pretrained Weights

EveNet is released as a pretrained foundation model. Start from these weights when fine-tuning or making predictions.

- Browse and download weights directly from HuggingFace:  
  👉 [Avencast/EveNet on HuggingFace](https://huggingface.co/Avencast/EveNet/tree/main)

- Place the downloaded `.ckpt` file somewhere accessible and update your YAML configs with the path (see below).


## 5. Prepare Input Data

Do not run the scripts under `preprocessing/`, which are only for large-scale pretraining. For fine-tuning, prepare your own dataset in the EveNet format.

You are responsible for converting your physics ntuples into the EveNet parquet + metadata format.  
See [data preparation guide](data_preparation.md) for details on schema, normalization, and writer options.  

> 🔍 Tip: Keep your preprocessing configs in version control so you can reproduce and document each dataset.


## 6. Configure an Experiment

> **Note:** The example configs are **not standalone**. Each one uses  
> `default: ...yaml` to load additional base configs. The parser resolves  
> these paths relative to the example’s location, so you must also copy  
> the referenced YAML files and preserve their directory structure.

1. Copy both `share/finetune-example.yaml` (for training) and `share/predict-example.yaml` (for inference) into your working directory.  

2. Update key fields for your experiment:
   - `platform.data_parquet_dir` → directory of your processed parquet files  
   - `Dataset.normalization_file` → path to the normalization statistics you created  
   - `options.Training.pretrain_model_load_path` → pretrained checkpoint to load  
   - `logger` → project name, WANDB API key, or local log directory  

3. For a detailed description of every section and all available overrides, see the [configuration reference](configuration.md).

## 7. Fine-Tune the Model

1. Export your Weights & Biases API key if you plan to log online.
   ```bash
   export WANDB_API_KEY=<your_key>
   ```
2. Launch training with your updated YAML.
   - **Quick start users:** run the packaged CLI after `pip install evenet`.
     ```bash
     evenet-train path/to/your-train-config.yaml
     ```
   - **Source checkout:** execute the module directly to pick up local code edits.
     ```bash
     python -m evenet.train path/to/your-train-config.yaml
     ```
3. Monitor progress:
   - Console output provides per-epoch metrics and checkpoint locations.
   - WANDB dashboards (if enabled) visualize loss curves and system stats.
   - Checkpoints and logs are stored under `options.Training.model_checkpoint_save_path`.


## 8. Generate Predictions

1. Ensure the prediction YAML points to your trained (or pretrained) checkpoint via `options.Training.model_checkpoint_load_path`.
2. Launch inference with either interface.
   ```bash
   # PyPI package
   evenet-predict path/to/your-predict-config.yaml

   # Source checkout
   python -m evenet.predict path/to/your-predict-config.yaml
   ```
3. Outputs land in the configured writers (e.g., parquet, numpy archives). See `docs/predict.md` for writer options and schema notes.


## 9. Explore and Iterate

- Use the artifacts written in the prediction step for downstream analysis (examples live under `downstreams/`).
- Adjust YAML hyperparameters, architecture templates, or preprocessing selections and repeat the workflow.
- When adding new datasets or modules, contribute documentation updates so the next user can follow your path.

Happy exploring! 🚀
