# EveNet 🌌

[![Open App](https://img.shields.io/badge/Open-Doc-blue?style=for-the-badge)](https://uw-epe-ml.github.io/EveNet_Public/)

EveNet is a multi-task, event-level neural network for large-scale high-energy physics analyses. It combines a Ray + PyTorch Lightning training loop, a flexible multi-GPU inferstructure for slurms, and modular YAML-driven configuration so new datasets and studies can be onboarded quickly.

---

![](docs/network_summary.png)

---

## 🚀 Installation

EveNet is now packaged as a Python module. Install it directly from PyPI (or a local checkout while developing) and the CLI entry points become available automatically:

```bash
pip install evenet
```

For local development you can install the current repository in editable mode:

```bash
pip install -e .
```

> **Note:** The package intentionally does not declare the full CUDA / GPU software stack as dependencies. Provision the execution environment yourself—either by using the Docker image under [`Docker/`](Docker/) or by installing the requirements from [`requirements.txt`](requirements.txt) / [`requirement_docker.txt`](requirement_docker.txt) on compatible hardware before launching the CLI.

## 🛠️ Command line interface

The package exposes training and prediction helpers so experiments can be launched without Python boilerplate. Provide the same YAML configuration files that power the existing workflows.

```bash
# Launch distributed training with Ray + Lightning
evenet-train path/to/config.yaml --ray_dir ~/ray_results

# Run inference with a trained checkpoint
evenet-predict path/to/predict_config.yaml
```

---

## 🤝 Contributing

Improvements are welcome! File an issue or open a pull request for bug fixes, new physics processes, or documentation tweaks. When you add new components or datasets, update the relevant markdown guides so future users can follow along easily.

