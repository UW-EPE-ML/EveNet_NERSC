# EveNet

EveNet is a multitask event-level neural network designed for large-scale high-energy physics analyses. The repository couples a Ray + PyTorch Lightning training loop with a preprocessing pipeline that turns NERSC-format ntuples into parquet shards, and a modular configuration system that enables fast experimentation across physics processes.

## Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare data** – convert raw ntuples into the parquet + metadata bundle expected by EveNet. Follow the [data preparation guide](docs/data_preparation.md).
3. **Fine-tune or train** – customize `share/finetune-example.yaml`, then launch training:
   ```bash
   WANDB_API_KEY=<your_key> \
   python evenet/train.py --config share/finetune-example.yaml
   ```
4. **Predict** – load a trained checkpoint with `share/predict-example.yaml`:
   ```bash
   python evenet/predict.py --config share/predict-example.yaml
   ```

## Repository Layout

| Path | Description |
| --- | --- |
| `evenet/` | Core Python package (Lightning engine, model, Ray data plumbing, utilities). |
| `preprocessing/` | Scripts for converting raw samples into model-ready parquet datasets plus normalization statistics. |
| `share/` | Example YAMLs for preprocessing, options, network architecture, event/process metadata, and resonance catalogs. |
| `docs/` | User-facing documentation (this README, data/config/model references). |
| `downstreams/` | Illustrative downstream analysis notebooks and scripts. |
| `NERSC/` & `Docker/` | Environment helpers for NERSC deployments and container builds. |

## Documentation Map

- [Data preparation & input reference](docs/data_preparation.md)
- [Configuration reference](docs/configuration.md)
- [Model architecture guide](docs/model_architecture.md)

Each document links back to the relevant YAML templates under `share/*-example.yaml`, making it straightforward to adapt EveNet to new datasets and tasks.

## Typical Workflow

1. **Generate parquet data** using the preprocessing CLI. Capture the produced `data_*.parquet`, `shape_metadata.json`, and `normalization.pt`.
2. **Clone the example configs** from `share/` and fill in absolute paths for parquet, normalization, and checkpoint locations.
3. **Run training** to fine-tune or continue pretraining on the prepared samples. Monitor metrics in Weights & Biases or the local log directory.
4. **Run prediction or downstream tasks** on Ray clusters or workstations by pointing to the trained checkpoint and matching normalization file.

## Contributing

Contributions are welcome! Please open issues or pull requests with improvements to the preprocessing pipeline, configuration templates, or documentation. When adding new features, update the relevant guides under `docs/` so new users can benefit from the changes.
