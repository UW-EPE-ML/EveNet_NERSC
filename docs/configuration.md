# Configuration Reference

EveNet composes its runtime configuration from modular YAML fragments. This document explains how the loader merges defaults, how to edit the example configs under `share/*-example.yaml`, and which sections control training, datasets, and physics metadata.

## Configuration Loader

`evenet/control/global_config.Config` reads a top-level YAML file (e.g., `share/finetune-example.yaml`) and merges each section with its declared `default` file. Entries are wrapped in a recursive `DotDict`, allowing attribute access (`config.options.Training.epochs`). Event metadata is instantiated as an `EventInfo` object so downstream code can look up feature schemas and resonance trees.【F:evenet/control/global_config.py†L1-L120】【F:evenet/control/global_config.py†L121-L199】

Key behaviors:

- A section with `default: path/to/base.yaml` loads the base file and merges any overrides that follow it.
- `event_info` and `resonance` sections are required; the loader constructs `EventInfo` using both blocks.
- Additional YAML fragments (e.g., `process_info`) are merged verbatim and exposed on `global_config` for preprocessing, training, and logging.【F:evenet/control/global_config.py†L68-L137】

## Top-Level Example (`share/finetune-example.yaml`)

The example fine-tuning config illustrates the major sections:

| Section | Purpose |
| --- | --- |
| `platform` | Ray cluster parameters (number of workers, GPU/CPU resources, batch size, prefetching). Also sets `data_parquet_dir` for dataset discovery.【F:share/finetune-example.yaml†L1-L23】 |
| `logger` | Optional Weights & Biases and local file logging configuration. The `run_name` anchor is reused between loggers.【F:share/finetune-example.yaml†L24-L38】 |
| `options` | Points to `options/options.yaml` for optimizer, scheduler, and component defaults, then overrides specific entries for this run (epochs, checkpoint paths, task inclusion).【F:share/finetune-example.yaml†L40-L107】 |
| `Dataset` | Path overrides for normalization statistics, dataset limits, and validation split boundaries.【F:share/finetune-example.yaml†L109-L118】 |
| `network` | Architecture template selection (`network/network-20M.yaml`) plus optional overrides (e.g., disabling feature drop).【F:share/finetune-example.yaml†L120-L133】 |
| `event_info`, `resonance` | Pointers to the canonical particle definitions used during preprocessing and model construction. You can extend or override subsections inline.【F:share/finetune-example.yaml†L135-L143】 |

For inference, `share/predict-example.yaml` reuses the same defaults but introduces an `options.prediction` block for the on-disk writer and requires `model_checkpoint_load_path` to point to a saved Lightning checkpoint.【F:share/predict-example.yaml†L1-L74】

### Platform Settings

| Key | Effect |
| --- | --- |
| `data_parquet_dir` | Directory containing the parquet shards and `shape_metadata.json`. |
| `number_of_workers` | Ray workers launched for training/prediction. |
| `resources_per_worker` | Dict specifying CPU/GPU requirements per worker. |
| `batch_size`, `prefetch_batches` | Passed directly to `iter_torch_batches` for Ray Data loaders. |
| `use_gpu` | Toggle GPU allocation when running on CPU-only environments. |

### Dataset Block

| Key | Meaning |
| --- | --- |
| `dataset_limit` | Fraction of available events to sample (1.0 uses the full dataset). |
| `normalization_file` | Absolute path to the `normalization.pt` produced during preprocessing. |
| `val_split` | Two-element list describing the validation window in `[0,1]` fraction of the shuffled dataset. |

## Options (`share/options/options.yaml`)

The options file stores optimizer, scheduler, and task toggles. Highlights include:

### Training

| Field | Description |
| --- | --- |
| `total_epochs` / `epochs` | Planned schedule vs. epochs executed in the current run. |
| `learning_rate`, `weight_decay` | Default hyperparameters broadcast to individual components via YAML anchors. |
| `model_checkpoint_*` | Save/load locations for Lightning checkpoints. |
| `diffusion_every_n_epochs`, `diffusion_every_n_steps` | Evaluation cadence for diffusion-based heads. |
| `apply_event_weight` | If `true`, multiplies losses by per-event weights from preprocessing.【F:share/options/options.yaml†L1-L71】 |

### Component Blocks

Each model component inherits defaults and can be toggled independently:

| Component | Notable knobs |
| --- | --- |
| `GlobalEmbedding`, `PET`, `ObjectEncoder` | Optimizer group, warm-up behavior, optional layer freezing strategies. |
| `Classification`, `Regression`, `Assignment`, `Segmentation` | `include` flag, loss scales, task-specific hyperparameters (e.g., focal gamma, dice loss weight). |
| `GlobalGeneration`, `ReconGeneration`, `TruthGeneration` | Diffusion step count, whether to reuse generated samples, optimizer routing.【F:share/options/options.yaml†L22-L159】 |

All sub-blocks share the same schema: learning rate, optimizer type, warm-up toggle, weight decay, and fine-grained freeze controls. This structure makes it easy to partially freeze modules or adjust loss weights during fine-tuning.

### Progressive Training

`ProgressiveTraining` defines staged schedules where loss weights and training parameters evolve across epochs. Each stage specifies an `epoch_ratio`, optional `transition_ratio`, and lists of `[start, end]` values for loss weights or parameters (e.g., `noise_prob`, `ema_decay`). Modify or disable stages to control curriculum learning during long runs.【F:share/options/options.yaml†L160-L218】

### EMA and Early Stopping

Inside the `Training` overrides (e.g., in `finetune-example.yaml`), the `EMA` block controls exponential moving average updates, including decay, start epoch, and whether to swap EMA weights into the model at the end of training. `EarlyStopping` follows Lightning’s API for patience, monitored metric, and comparison mode.【F:share/finetune-example.yaml†L60-L107】

## Network Templates (`share/network/*.yaml`)

Network YAMLs describe architectural hyperparameters for the body and heads. For instance, `network-20M.yaml` configures transformer depth, attention heads, local neighborhood size, and diffusion latent dimensions. Override specific fields under the `network` section of your top-level config to experiment without modifying the base file.【F:share/finetune-example.yaml†L120-L133】

## Physics Metadata

### Event Info (`share/event_info/multi_process.yaml`)

Defines:

- **Inputs** – sequential particle features (`energy`, `p_T`, `η`, `φ`, b-tag, lepton flag, charge) and global event conditions (MET, multiplicities, aggregate masses). Each entry also specifies the normalization mode applied during preprocessing and model inference.【F:share/event_info/multi_process.yaml†L1-L35】
- **Resonance topology** – `EVENT` lists parent particles and their daughters for each process. These relationships drive assignment targets and segmentation tags.【F:share/event_info/multi_process.yaml†L37-L120】
- **Regression and segmentation targets** – enumerations of momenta to regress and particle groups for segmentation heads.【F:share/event_info/multi_process.yaml†L398-L440】

### Process Info (`share/process_info/default.yaml`)

Captures per-process metadata:

- `category` (used for logging or grouping processes).
- `process_id` (integer label stored alongside events).
- `diagram` definitions referencing reusable resonances (with symmetry hints). This mapping is consumed during preprocessing when assigning truth particles and computing weights.【F:share/process_info/default.yaml†L1-L64】

You can duplicate `default.yaml` to create experiment-specific subsets or rename processes via the `rename_to` key (see entries under `share/process_info/pretrain.yaml`). Ensure the same file is used during preprocessing and training to keep class indices aligned.

## Prediction Extras

When running inference the `options.prediction` block specifies the output directory, filename, and optional lists of extra features to persist. The `platform` section mirrors training but typically runs with `num_workers` tuned for evaluation throughput. Set `Training.model_checkpoint_load_path` to the checkpoint you wish to score.【F:share/predict-example.yaml†L1-L74】

## Adding a New Experiment

1. Copy `share/finetune-example.yaml` (or `predict-example.yaml` for inference) to a new file.
2. Update `platform.data_parquet_dir`, `options.Dataset.normalization_file`, and checkpoint paths.
3. Toggle heads or adjust hyperparameters under `options.Training` and `options.Training.Components` as needed.
4. If introducing new physics processes, extend the appropriate YAMLs under `share/event_info/`, `share/process_info/`, and rerun preprocessing so the new schema is reflected everywhere.

Following this structure keeps EveNet’s configuration reproducible and shareable across collaborators.
