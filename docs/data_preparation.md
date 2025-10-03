# Data Preparation & Input Reference

This guide walks through converting raw ntuples into EveNet-ready parquet shards and explains the tensors stored in each dataset. The preprocessing stack lives in `preprocessing/` and is driven by the YAML template `share/preprocess_pretrain.yaml`.

## Overview of the Pipeline

1. **Configure preprocessing** – copy `share/preprocess_pretrain.yaml` and customize input directories, selection definitions, and output paths.
2. **Run the CLI** – execute `preprocessing/preprocess.py` to iterate over raw runs, convert them into the canonical tensors, and accumulate normalization statistics.
3. **Collect artifacts** – the script emits a shuffled parquet file per run plus shared metadata (`shape_metadata.json`, `normalization.pt`, cutflow reports) needed for training and evaluation.

## Command-Line Usage

```bash
python preprocessing/preprocess.py share/preprocess_pretrain.yaml \
  --in_dir /path/to/Run_XXX \
  --store_dir /path/to/output \
  --cpu_max 32
```

Key switches:

| Argument | Description |
| --- | --- |
| `preprocess_config` | Path to the YAML file describing selections and process metadata (default: `share/preprocess_pretrain.yaml`). |
| `--in_dir` | Directory containing a single run (e.g., `Run_2.Dec20/run_yulei_13`). Required when `--pretrain_dirs` is not provided. |
| `--pretrain_dirs` | Optional list of parent directories. Each immediate subdirectory is processed in parallel. Forces the two-level folder structure used in NERSC campaigns. |
| `--store_dir` | Destination folder for parquet shards and metadata (`Storage/` by default). |
| `--cpu_max` | Maximum number of CPU cores for multiprocessing. |

The script can also process multiple campaign roots in parallel:

```bash
python preprocessing/preprocess.py share/preprocess_pretrain.yaml \
  --pretrain_dirs /nersc/campaignA /nersc/campaignB \
  --store_dir /workspace/datasets/evenet \
  --cpu_max 48
```

## Preprocessing Configuration (`share/preprocess_pretrain.yaml`)

### Global Settings

| Key | Purpose |
| --- | --- |
| `max_neutrinos` | Maximum number of invisible particles stored per event. Controls padding for the neutrino generation head. |

### Selection Aliases and Anchors

Aliases map human-friendly names to raw array paths so that selection expressions remain readable. Anchors provide reusable boolean expressions for common kinematic requirements.

| Alias | Maps to |
| --- | --- |
| `n_lep` | `INPUTS/Conditions/nLepton` |
| `n_bjet` | `INPUTS/Conditions/nbJet` |
| `n_jet` | `INPUTS/Conditions/nJet` |
| `is_lep` | `INPUTS/Source/isLepton` |
| `is_bjet` | `INPUTS/Source/btag` |
| `is_valid` | `INPUTS/Source/MASK` |
| `pt` | `INPUTS/Source/pt` |

Representative anchors:

| Anchor | Expression |
| --- | --- |
| `lep1_pT_sel` | Leading lepton transverse momentum > 10 GeV. |
| `bjet2_pT_sel` | Second-leading b jet transverse momentum > 20 GeV. |
| `jet4_pT_sel` | Fourth jet transverse momentum > 20 GeV (non-b, non-lepton). |

Anchors are composed in the `selections` block to define process-specific filters (e.g., `TT1L`, `ttW_FullHadronics`). Each selection becomes a list of NumPy expressions evaluated against the aliases, and the resulting masks drive the cutflow recorded per process.【F:share/preprocess_pretrain.yaml†L1-L75】【F:share/preprocess_pretrain.yaml†L76-L134】

### Outputs and Reports

Running the pipeline produces the following files inside `store_dir`:

| Artifact | Description |
| --- | --- |
| `data_<run>.parquet` | Flattened event table (one per processed run) containing all tensors listed below.【F:preprocessing/preprocess.py†L262-L299】 |
| `shape_metadata.json` | Original tensor shapes for each parquet column, used to reconstruct arrays when reading the dataset.【F:preprocessing/preprocess.py†L300-L309】 |
| `normalization.pt` | Torch dictionary with feature means/standard deviations, class balances, and diffusion statistics consumed during training.【F:preprocessing/postprocessor.py†L360-L406】 |
| `cutflow_summary.txt`, `cutflows.json` | Human-readable and JSON summaries of selection efficiencies for each process.【F:preprocessing/preprocess.py†L420-L440】 |

## Dataset Layout

Each row in the parquet file corresponds to a single event. The following tensors are generated before flattening and can be reconstructed using `shape_metadata.json`:

### Inputs (`INPUTS/*`)

| Tensor | Shape | Description |
| --- | --- | --- |
| `INPUTS/Source` features | `(N_events, N_particles, 7)` | Sequential particle attributes: energy, `p_T`, `η`, `φ`, b-tag, lepton flag, charge. Masks stored in `INPUTS/Source/MASK`.【F:preprocessing/preprocess.py†L220-L257】【F:share/event_info/multi_process.yaml†L10-L35】 |
| `INPUTS/Conditions` | `(N_events, 9)` | Global event-level scalars (MET, multiplicities, summed energies/masses). Mask stored in `conditions_mask`.【F:preprocessing/preprocess.py†L220-L257】【F:share/event_info/multi_process.yaml†L24-L35】 |
| `num_vectors` | `(N_events,)` | Total object count per event (sequential + global).【F:preprocessing/preprocess.py†L220-L257】 |
| `num_sequential_vectors` | `(N_events,)` | Number of valid sequential objects (particle-flow candidates).【F:preprocessing/preprocess.py†L220-L257】 |

### Supervision Targets

| Tensor | Shape | Description |
| --- | --- | --- |
| `classification` | `(N_events,)` | Process label encoded per `share/process_info/*.yaml`, weighted during training via `event_weight`.【F:preprocessing/preprocess.py†L220-L287】 |
| `assignments-indices` | `(N_events, N_targets, N_daughters)` | Indices mapping reconstructed particles to truth daughters for each resonance. Companion mask tensors `assignments-mask` and `assignments-indices-mask` mark valid entries.【F:preprocessing/evenet_data_converter.py†L64-L122】 |
| `regression-data` | `(N_events, N_regression)` | Continuous targets (momenta) per process/particle with boolean mask `regression-mask`.【F:preprocessing/evenet_data_converter.py†L124-L190】 |
| `segmentation-class` | `(N_events, N_segments, N_tags)` | One-hot membership for resonance-specific particle groups. Additional arrays include `segmentation-data`, `segmentation-momentum`, and `segmentation-full-class`.【F:preprocessing/evenet_data_converter.py†L92-L163】 |
| `x_invisible` | `(N_events, max_neutrinos, F)` | Invisible particle features (padded) with masks tracking raw and valid counts. Controlled by `max_neutrinos`.【F:preprocessing/preprocess.py†L232-L247】 |

### Additional Metadata

| Tensor | Shape | Description |
| --- | --- | --- |
| `subprocess_id` | `(N_events,)` | Integer identifier for the source subprocess, matching the ordering in `event_info.event_mapping`.【F:preprocessing/preprocess.py†L220-L299】 |
| `event_weight` | `(N_events,)` | Sample weights derived from process metadata, propagated to normalization statistics and balances.【F:preprocessing/preprocess.py†L246-L268】 |

## Reading the Dataset

During training or prediction the helper `prepare_datasets` in `shared.py` reads the parquet shards, reshapes columns using `shape_metadata.json`, and applies the normalization tensors saved in `normalization.pt`. Always ensure that:

- `platform.data_parquet_dir` points to the folder containing the parquet shards.
- `options.Dataset.normalization_file` points to the matching `normalization.pt` file.
- The event/process YAMLs used for preprocessing are the same ones supplied in the runtime config, so feature ordering and masks remain consistent.【F:preprocessing/postprocessor.py†L360-L406】【F:share/finetune-example.yaml†L1-L86】

With these artifacts in place, EveNet can ingest datasets across Ray workers and reproduce the exact tensor layout defined by the preprocessing configuration.
