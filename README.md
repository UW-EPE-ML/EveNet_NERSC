# EveNet 🌌

[![Open App](https://img.shields.io/badge/Open-Doc-blue?style=for-the-badge)](https://uw-epe-ml.github.io/EveNet_Public/)


EveNet is a pretrained, multi-task foundation model for event-level collider physics. 
It comes with a scalable Ray + PyTorch Lightning training pipeline, SLURM-ready multi-GPU infrastructure, 
and modular YAML configuration, 
so researchers can quickly fine-tune it on their own datasets and extend it to new physics analyses.

---

![](docs/network_summary.png)

---

## 🚀 Pretrained Weights

Start directly from pretrained EveNet checkpoints for fine-tuning or inference:

👉 HuggingFace: [Avencast/EveNet](https://huggingface.co/Avencast/EveNet/tree/main)

---

## 📦 Python Package Distribution

The repository now ships as a lightweight Python package that exposes ready-to-run CLI entry points for
training and prediction. We intentionally **do not declare runtime dependencies** in `pyproject.toml`
because EveNet targets GPU-enabled environments that often require bespoke CUDA, PyTorch, and Ray
builds. Create your own virtual environment on **Python 3.12+** or use one of the provided Docker images
before installing EveNet.

```bash
pip install .
```

After installation you can launch the existing Ray/Lightning workflows directly from the command line:

```bash
# Start a training run
evenet-train share/configs/train.yaml --ray_dir ~/ray_results

# Run inference
evenet-predict share/configs/predict.yaml
```

Both CLIs expect the same YAML configuration files documented in [`docs/train.md`](docs/train.md) and
[`docs/predict.md`](docs/predict.md). Ensure your environment has access to GPUs (where required) and
the appropriate dataset shards referenced in the configuration.

---

## 🤝 Contributing

Improvements are welcome! File an issue or open a pull request for bug fixes, new physics processes, or documentation tweaks. When you add new components or datasets, update the relevant markdown guides so future users can follow along easily.
