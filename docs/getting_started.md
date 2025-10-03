# 📘 EveNet Tutorial: From Zero to First Predictions

New to EveNet? This tutorial walks you through the end-to-end workflow so you can move from a clean checkout to model predictions with confidence. Each step links to a deeper reference guide if you need more detail.

---

## 1. Understand the Project Layout

Before running any commands, skim these key directories:

- `evenet/` – PyTorch Lightning modules, Ray data pipelines, and trainer utilities.
- `share/` – Ready-to-edit YAML configurations for fine-tuning and prediction.
- `docs/` – Reference documentation that expands on this tutorial. Start with [Model Architecture Tour](model_architecture.md) to see how point-cloud and global features flow through EveNet.
- `downstreams/` – Example downstream analysis scripts built on top of EveNet outputs.

---

## 2. Set Up Your Environment

1. **Pull the official Docker image (recommended).** It bundles CUDA, PyTorch, and all Python requirements so you can start immediately.
   ```bash
   docker pull docker.io/avencast1994/evenet:1.3
   docker run --gpus all -it \
     -v /path/to/your/data:/workspace/data \
     -v $(pwd):/workspace/project \
     docker.io/avencast1994/evenet:1.3
   ```
   Inside the container, switch to `/workspace/project` to use your local checkout. If Docker is unavailable, install dependencies manually with `pip install -r requirements.txt` on Python 3.10+.
2. **Review cluster helpers as needed.** The `Docker/` and `NERSC/` directories include recipes and SLURM launch scripts tailored for HPC environments.
3. **Verify GPU visibility (if available).**
   ```bash
   python -c "import torch; print(torch.cuda.device_count())"
   ```

---

## 3. Download Pretrained Weights

EveNet is released as a pretrained foundation model. Start from these weights when fine-tuning or making predictions.

- Browse and download weights directly from HuggingFace:  
  👉 [Avencast/EveNet on HuggingFace](https://huggingface.co/Avencast/EveNet/tree/main)

- Place the downloaded `.ckpt` file somewhere accessible and update your YAML configs with the path (see below).

---

## 4. Prepare Input Data

Do not run the scripts under `preprocessing/`, which are only for large-scale pretraining. For fine-tuning, prepare your own dataset in the EveNet format.

You are responsible for converting your physics ntuples into the EveNet parquet + metadata format.  
See [data preparation guide](data_preparation.md) for details on schema, normalization, and writer options.  

> 🔍 Tip: Keep your preprocessing configs in version control so you can reproduce and document each dataset.

---

## 5. Configure an Experiment

1. Copy `share/finetune-example.yaml` (for training) and `share/predict-example.yaml` (for inference) into a working directory.
2. Update paths and experiment metadata:
   - `platform.data_parquet_dir` → location of your processed parquet files
   - `Dataset.normalization_file` → normalization statistics file you created
   - `options.Training.pretrain_model_load_path` → path to downloaded pretrained weights
   - `logger` → project names, WANDB API key, or local log paths
3. Review the [configuration reference](configuration.md) for a description of every YAML section and available overrides.

---

## 6. Fine-Tune the Model

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

---

## 7. Generate Predictions

1. Ensure the prediction YAML points to your trained (or pretrained) checkpoint via `options.Training.model_checkpoint_load_path`.
2. Launch inference.
   ```bash
   python evenet/predict.py path/to/your-predict-config.yaml
   ```
3. Outputs land in the configured writers (e.g., parquet, numpy archives). See `docs/predict.md` for writer options and schema notes.

---

## 8. Explore and Iterate

- Use the artifacts written in the prediction step for downstream analysis (examples live under `downstreams/`).
- Adjust YAML hyperparameters, architecture templates, or preprocessing selections and repeat the workflow.
- When adding new datasets or modules, contribute documentation updates so the next user can follow your path.

Happy exploring! 🚀
