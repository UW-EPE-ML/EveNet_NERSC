# EveNet Documentation Portal

Welcome to the EveNet knowledge base! This site hosts the same Markdown guides that live in the repository, but it formats them with navigation, search, and table-of-contents support so new collaborators can onboard quickly.

Use the navigation menu to jump straight to the guide you need, or start with the highlights below.

## 🚀 Quick Starts

- **Plug-and-play setup?** Follow the Quick Start path in the [Getting Started tutorial](getting_started.md) to pair the official Docker image with the PyPI package.
- **Hacking on the source?** The same guide outlines the advanced workflow for cloning the repo and running modules directly.
- **Prepping datasets?** Head to the [Data Preparation guide](data_preparation.md) to learn how to configure preprocessing YAMLs and generate parquet shards.
- **Training & inference.** Consult the [Training playbook](train.md) and [Prediction walkthrough](predict.md) for command-line examples and Ray configuration tips.

## 🧠 Reference Library

- **Architecture deep dive.** The [Model Architecture overview](model_architecture.md) explains the hybrid point-cloud and global feature encoders that power EveNet.
- **Configuration catalogue.** The [Configuration reference](configuration.md) documents every YAML option and how they interact.
- **Internal utilities.** Project maintainers can review [internal preprocessing notes](preprocess_internal_only.md) for NERSC-specific helpers.

## 🔄 Keeping Docs in Sync

The MkDocs configuration (`mkdocs.yml`) reads directly from the `docs/` folder, so improvements to the Markdown files automatically flow to the website. When you update a guide, commit the change and the GitHub Pages workflow will rebuild the site.

Looking to preview locally? See the README for the commands to install MkDocs and run `mkdocs serve`.
