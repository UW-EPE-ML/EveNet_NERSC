# EveNet 🌌

EveNet is a multi-task, event-level neural network for large-scale high-energy physics analyses. It combines a Ray + PyTorch Lightning training loop, a flexible multi-GPU inferstructure for slurms, and modular YAML-driven configuration so new datasets and studies can be onboarded quickly.

---

## 🧭 Quick Navigation
- 👉 [Data preparation & input guide](docs/data_preparation.md)
- ⚙️ [Configuration reference](docs/configuration.md)
- 🧠 [Model architecture tour](docs/model_architecture.md)

---

## 🚀 Quickstart Workflow

| Step | Action | Command / Notes |
| --- | --- | --- |
| 1️⃣ | Install dependencies | ```bash
pip install -r requirements.txt
``` |
| 2️⃣ | Prepare your dataset | Follow the [data guide](docs/data_preparation.md#run-the-preprocessing-cli) to configure preprocessing for your ntuples, then build parquet shards + normalization stats. |
| 3️⃣ | Launch training | Tweak the example YAML (see the [configuration reference](docs/configuration.md)) and run:<br>`WANDB_API_KEY=<your_key> \`<br>`python evenet/train.py share/finetune-example.yaml` |
| 4️⃣ | Run prediction | Point the prediction YAML at your checkpoint and execute:<br>`python evenet/predict.py share/predict-example.yaml` |
| 5️⃣ | Explore results | Visualize metrics in Weights & Biases or the local log directory listed in the YAML. |

> 💡 **Tip:** Ray launches one worker per GPU/CPU pair by default. Adjust `platform.number_of_workers` and `platform.resources_per_worker` inside the YAML to scale up or down.

---

## 🗂️ Repository Highlights

| Path | What lives here? |
| --- | --- |
| `evenet/` | Core Python package: Lightning engine, Ray data adapters, model modules, utilities. |
| `preprocessing/` | Scripts that turn raw samples into parquet shards, metadata, and normalization statistics. |
| `share/` | Example YAML configurations (`*-example.yaml`) plus reusable templates under `options/`, `network/`, `event_info/`, `process_info/`, and `resonance/`. |
| `docs/` | User documentation (this README plus deep dives on data, configs, and architecture). |
| `downstreams/` | Example analyses demonstrating how to consume EveNet outputs. |
| `NERSC/`, `Docker/` | Environment helpers for HPC deployments and container builds. |

---

## 🏁 End-to-End Checklist

1. 📦 **Package setup** – install requirements and verify CUDA/Ray availability.
2. 🧪 **Preprocessing dry run** – process a small run to confirm parquet + metadata generation.
3. 🧾 **Config audit** – update dataset paths, logging directories, and checkpoint destinations in the example YAMLs.
4. 🧉 **Training run** – start with a short epoch count (`options.Training.epochs`) to validate metrics/logging.
5. 🛰️ **Prediction or downstream analysis** – reuse the same normalization + event metadata to ensure tensors line up.

---

## 🤝 Contributing

Improvements are welcome! File an issue or open a pull request for bug fixes, new physics processes, or documentation tweaks. When you add new components or datasets, update the relevant markdown guides so future users can follow along easily.

