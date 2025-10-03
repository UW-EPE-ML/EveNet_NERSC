# ⚙️ Configuration Reference

Welcome to the EveNet control center! This document explains how the YAML examples under `share/*-example.yaml` fit together, how defaults are merged, and which knobs to tweak for your experiments.

- [Loader basics](#loader-basics)
- [Tour of `share/finetune-example.yaml`](#finetune-example)
- [Options deep dive](#options-deep-dive)
- [Network templates](#network-templates)
- [Physics metadata](#physics-metadata)
- [Prediction-specific notes](#prediction-notes)
- [Creating a new experiment](#new-experiment)

---

## 🧰 Loader Basics {#loader-basics}

`evenet/control/global_config.py` reads a top-level YAML file (for example, `share/finetune-example.yaml`) and merges each section with its declared `default` template. Fields are exposed through a recursive `DotDict`, so you can write `config.options.Training.epochs` inside Python. Event metadata is instantiated as an `EventInfo` object, which downstream modules use to look up feature schemas and resonance trees. Peek at the implementation here: [`evenet/control/global_config.py`](../evenet/control/global_config.py).

**Key behaviors**

- `default: path/to/base.yaml` loads the base template before applying inline overrides.
- The `event_info` and `resonance` sections are required; the loader constructs `EventInfo` using both blocks.
- Additional sections (such as `process_info`) are merged verbatim and made available across preprocessing, training, and logging.

> 📝 **Remember:** the positional argument to `evenet/train.py` or `evenet/predict.py` is the path to your top-level YAML. No `--config` flag is necessary.

---

## 📄 Tour of `share/finetune-example.yaml` {#finetune-example}

This example highlights the main sections you will encounter:

| Section | What it controls |
| --- | --- |
| `platform` | Ray cluster parameters: worker count, per-worker CPU/GPU resources, batch size, and the `data_parquet_dir` used to discover shards. |
| `logger` | Logging destinations, including Weights & Biases and the optional local logger. Anchors let you reuse names across loggers. |
| `options` | Points to `options/options.yaml` for defaults, then overrides epochs, checkpoints, and component toggles. |
| `Dataset` | Dataset-wide overrides like `normalization_file`, dataset subsampling, and validation split. |
| `network` | Which architecture template to load (`network/network-20M.yaml`, for example) plus any inline tweaks. |
| `event_info`, `resonance` | Canonical particle definitions, resonance trees, and symmetries used during preprocessing and by the model. |

For inference, `share/predict-example.yaml` mirrors the structure but requires `Training.model_checkpoint_load_path` and defines an `options.prediction` block for output writers.

### Platform Keys

| Key | Description |
| --- | --- |
| `data_parquet_dir` | Folder containing `data_*.parquet` and `shape_metadata.json`. |
| `number_of_workers` | Ray workers to launch. Adjust alongside `resources_per_worker`. |
| `resources_per_worker` | CPU/GPU allocation per worker (e.g., `{CPU: 1, GPU: 1}`). |
| `batch_size`, `prefetch_batches` | Passed to Ray Data’s `iter_torch_batches` loader. |
| `use_gpu` | Toggle GPU usage for CPU-only runs. |

### Dataset Block

| Key | Meaning |
| --- | --- |
| `dataset_limit` | Fraction of the shuffled dataset to use (1.0 = full dataset). |
| `normalization_file` | Absolute path to `normalization.pt` from preprocessing. |
| `val_split` | `[start, end]` fraction describing the validation window. |

---

## 🧩 Options Deep Dive {#options-deep-dive}

`share/options/options.yaml` collects optimizer groups, scheduler defaults, and component toggles. Use the top-level `options` block in your example file to override any field.

### Training Settings

| Field | Description |
| --- | --- |
| `total_epochs` / `epochs` | Planned total vs. epochs executed in the current run. |
| `learning_rate`, `weight_decay` | Shared defaults broadcast to component blocks. |
| `model_checkpoint_save_path` | Directory for Lightning checkpoints. |
| `model_checkpoint_load_path` | Resume training from this checkpoint (set to `null` to start fresh). |
| `pretrain_model_load_path` | Load weights for fine-tuning without resuming optimizer state. |
| `diffusion_every_n_epochs`, `diffusion_every_n_steps` | Validation cadence for diffusion-based heads. |
| `eval_metrics_every_n_epochs` | Controls how often expensive metrics are computed. |
| `apply_event_weight` | When `true`, multiplies losses by per-event weights from preprocessing. |

### Component Blocks

Each component inherits optimizer settings and can be toggled independently:

| Component | Notable knobs |
| --- | --- |
| `GlobalEmbedding`, `PET`, `ObjectEncoder` | Optimizer group, warm-up behavior, layer freezing. |
| `Classification`, `Regression`, `Assignment`, `Segmentation` | `include` flags, loss scales, attention/decoder depth, mask usage. |
| `GlobalGeneration`, `ReconGeneration`, `TruthGeneration` | Diffusion steps, reuse settings, optimizer routing. |

### Progressive Training

The `ProgressiveTraining` section defines staged curricula. Each stage supplies an `epoch_ratio`, optional `transition_ratio`, and start/end values for parameters (loss weights, noise probabilities, EMA decay). Modify stages to ramp tasks gradually or disable them for single-stage training.

### EMA & Early Stopping

Within `options.Training` you will find:

- `EMA` – enable/disable exponential moving averages, set decay, start epoch, and whether to swap EMA weights into the model after loading or at run completion.
- `EarlyStopping` – Lightning-compatible patience, monitored metric, comparison mode, and verbosity.

---

## 🏗️ Network Templates {#network-templates}

Files under `share/network/` specify architectural hyperparameters for the shared body and task heads. For example, [`share/network/network-20M.yaml`](../share/network/network-20M.yaml) defines transformer depth, attention heads, neighborhood sizes, and diffusion dimensions. Override specific fields inside the `network` section of your top-level YAML to experiment without editing the base template.

> 🧠 Combine template overrides with component toggles: disable `Body.PET.feature_drop` during fine-tuning, or shrink head hidden sizes for small datasets.

---

## 🧬 Physics Metadata {#physics-metadata}

### Event Info

[`share/event_info/multi_process.yaml`](../share/event_info/multi_process.yaml) describes:

- **Inputs** – sequential particle features and global condition vectors, including normalization strategies.
- **Resonance topology** – parent/daughter relationships for each process, powering assignments and segmentation targets.
- **Regression & segmentation catalogs** – enumerations of momenta and particle groups used by the respective heads.

### Process Info

[`share/process_info/default.yaml`](../share/process_info/default.yaml) captures per-process metadata:

- `category` names for grouping.
- `process_id` integers stored with each event.
- `diagram` entries referencing resonance definitions and symmetry hints.
- Optional `rename_to` fields for remapping process names during preprocessing.

Ensure the same event/process files are used in preprocessing **and** training so label ordering stays consistent.

### Resonance Catalogs

Templates under [`share/resonance/`](../share/resonance) hold reusable resonance definitions. Reference them from `event_info` to avoid duplicating decay trees across configs.

---

## 📦 Prediction Extras {#prediction-notes}

`share/predict-example.yaml` includes an `options.prediction` block where you can specify the output directory, filename, and any extra tensors to persist. The `platform` section mirrors training but is often tuned for throughput (e.g., more workers with smaller batch sizes). Always set `Training.model_checkpoint_load_path` to the checkpoint you want to score.

---

## 🆕 Creating a New Experiment {#new-experiment}

1. **Copy an example** – duplicate `share/finetune-example.yaml` (or `share/predict-example.yaml`).
2. **Update paths** – fill in `platform.data_parquet_dir`, `options.Dataset.normalization_file`, and logging/checkpoint directories.
3. **Toggle components** – adjust `options.Training.Components` to match the supervision you have.
4. **Tweak network settings** – override fields in the `network` section as needed.
5. **Track metadata** – if you add new processes or features, update the relevant YAMLs under `share/event_info/` and `share/process_info/`, then rerun preprocessing so tensors stay aligned.

Happy experimenting! 🧪

