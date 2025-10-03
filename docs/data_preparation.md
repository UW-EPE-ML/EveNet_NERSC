# 🧪 Data Preparation & Input Reference

Welcome! This guide shows how to convert raw ntuples into EveNet-ready parquet shards **and** how to interpret every tensor stored in the dataset. Keep it handy while customizing selections or debugging inputs.

- [Pipeline overview](#pipeline-overview)
- [⚙️ Run the preprocessing CLI](#run-the-preprocessing-cli)
- [📁 Outputs & artifact map](#outputs-artifact-map)
- [🧬 Dataset layout](#dataset-layout)
- [📚 Reading datasets in EveNet](#reading-datasets)

---

<a id="pipeline-overview"></a>
## 🔭 Pipeline Overview

1. **Draft a config** – start from your own YAML (you can peek at the internal `share/preprocess_pretrain.yaml` for inspiration) and list the processes, selections, and padding choices you need.
2. **Customize selections** – edit aliases, anchors, and selection blocks to match the processes you want to keep.
3. **Run the CLI** – point to the raw campaign directories and let the script build parquet shards, normalization stats, and cutflow summaries.
4. **Train or predict** – plug the generated artifacts into the example training or prediction YAMLs (see the [configuration reference](configuration.md)).

> ✨ **Pro tip:** keep preprocessing configs under version control so you can trace how a dataset was produced.

---

<a id="run-the-preprocessing-cli"></a>
## ⚙️ Run the Preprocessing CLI

```bash
python preprocessing/preprocess.py share/preprocess_pretrain.yaml \
  --in_dir /path/to/Run_XXX \
  --store_dir /path/to/output \
  --cpu_max 32
```

| Switch | Purpose |
| --- | --- |
| `preprocess_config` | Positional argument pointing to your YAML (defaults to `share/preprocess_pretrain.yaml`). |
| `--in_dir` | Directory with a single run (e.g., `Run_2.Dec20/run_yulei_13`). Required unless `--pretrain_dirs` is set. |
| `--pretrain_dirs` | Optional list of campaign roots; each immediate subdirectory is processed in parallel using the NERSC-style layout. |
| `--store_dir` | Destination for parquet shards, metadata, and cutflow summaries (defaults to `Storage/`). |
| `--cpu_max` | Cap the number of CPU cores used for multiprocessing. |

To process multiple campaigns concurrently:

```bash
python preprocessing/preprocess.py share/preprocess_pretrain.yaml \
  --pretrain_dirs /nersc/campaignA /nersc/campaignB \
  --store_dir /workspace/datasets/evenet \
  --cpu_max 48
```

### YAML Highlights

| Block | What it controls |
| --- | --- |
| `max_neutrinos` | Padding length for invisible particle features. |
| `selections.aliases` | Human-readable shortcuts for raw array names (e.g., `n_lep → INPUTS/Conditions/nLepton`). |
| `selection_anchors` | Reusable boolean expressions ("lep1_pT_sel" etc.) assembled into full selections. |
| `selections.<process>` | Ordered list of expressions evaluated on each event to build cutflows. |

Peek at the in-repo reference (`share/preprocess_pretrain.yaml`) when you need more examples of anchors or expressions.

---

<a id="outputs-artifact-map"></a>
## 📁 Outputs & Artifact Map

Running the pipeline fills `--store_dir` with a tidy bundle:

| Artifact | Description |
| --- | --- |
| `data_<run>.parquet` | Flattened event table per run containing all tensors listed below. See the writer in [`preprocessing/preprocess.py`](../preprocessing/preprocess.py#L361-L378). |
| `shape_metadata.json` | Original tensor shapes so EveNet can unflatten arrays on load. Generated in [`preprocessing/preprocess.py`](../preprocessing/preprocess.py#L370-L371). |
| `normalization.pt` | Torch dictionary with feature means/stds, class balances, and diffusion stats. Produced in [`preprocessing/postprocessor.py`](../preprocessing/postprocessor.py#L360-L406). |
| `cutflow_summary.txt`, `cutflows.json` | Human-readable and machine-readable summaries of selection efficiencies. Written near the end of [`preprocessing/preprocess.py`](../preprocessing/preprocess.py#L435-L440). |

> 🧾 Keep `shape_metadata.json` and `normalization.pt` alongside the parquet shards—training and prediction both rely on them.

---

<a id="dataset-layout"></a>
## 🧬 Dataset Layout

Each parquet row is a single event. Shapes below reference the **unflattened** tensors reconstructed with `shape_metadata.json`.

### Inputs (`INPUTS/*`)

| Tensor | Shape | Description |
| --- | --- | --- |
| `INPUTS/Source` | `(events, particles, 7)` | Particle-level features: energy, `pT`, `η`, `φ`, b-tag, lepton flag, charge. Mask stored as `INPUTS/Source/MASK`. Defined by the event metadata documented in the [configuration reference](configuration.md#physics-metadata). |
| `INPUTS/Conditions` | `(events, 9)` | Event-level scalars (MET, multiplicities, sums). Mask stored as `conditions_mask`. See the same event metadata section in the [configuration reference](configuration.md#physics-metadata). |
| `num_vectors` | `(events,)` | Total object count (global + sequential). Computed during preprocessing in [`preprocessing/preprocess.py`](../preprocessing/preprocess.py#L220-L257). |
| `num_sequential_vectors` | `(events,)` | Count of valid sequential objects. Set alongside `num_vectors` in [`preprocessing/preprocess.py`](../preprocessing/preprocess.py#L220-L257). |

### Supervision Targets

| Tensor | Shape | Description |
| --- | --- | --- |
| `classification` | `(events,)` | Process label encoded using the process metadata introduced in the [configuration reference](configuration.md#physics-metadata). Generated with event weights in [`preprocessing/preprocess.py`](../preprocessing/preprocess.py#L232-L287). |
| `assignments-indices` | `(events, targets, daughters)` | Maps reconstructed particles to truth daughters. Companion masks (`assignments-mask`, `assignments-indices-mask`) flag valid entries. Produced in [`preprocessing/evenet_data_converter.py`](../preprocessing/evenet_data_converter.py#L64-L122). |
| `regression-data` | `(events, regressions)` | Continuous targets (momenta, masses) with boolean mask `regression-mask`. Created in [`preprocessing/evenet_data_converter.py`](../preprocessing/evenet_data_converter.py#L124-L190). |
| `segmentation-class` | `(events, segments, tags)` | One-hot membership for resonance-specific particle groups. Extra tensors (`segmentation-data`, `segmentation-momentum`, `segmentation-full-class`) track auxiliary supervision. See [`preprocessing/evenet_data_converter.py`](../preprocessing/evenet_data_converter.py#L92-L163). |
| `x_invisible` | `(events, max_neutrinos, features)` | Invisible particle features (e.g., neutrinos) padded to `max_neutrinos`. Masks capture raw and valid counts inside [`preprocessing/preprocess.py`](../preprocessing/preprocess.py#L232-L247). |

### Additional Metadata

| Tensor | Shape | Description |
| --- | --- | --- |
| `subprocess_id` | `(events,)` | Integer identifier for the subprocess, matching the ordering described in the event metadata section of the [configuration reference](configuration.md#physics-metadata). |
| `event_weight` | `(events,)` | Per-event sample weight derived from process metadata. Computed where events are written in [`preprocessing/preprocess.py`](../preprocessing/preprocess.py#L246-L268). |

---

<a id="reading-datasets"></a>
## 📚 Reading Datasets in EveNet

When you run training or prediction, `evenet/shared.py` stitches everything back together:

1. **Read parquet shards** from `platform.data_parquet_dir`.
2. **Unflatten tensors** using `shape_metadata.json`.
3. **Apply normalization** from `options.Dataset.normalization_file`.
4. **Verify metadata** – use the same `event_info`/`process_info` YAML files that were active during preprocessing.

> ✅ Consistency checklist:
> - `platform.data_parquet_dir` → folder with `data_*.parquet` and `shape_metadata.json`.
> - `options.Dataset.normalization_file` → matching `normalization.pt`.
> - `event_info` + `process_info` in the runtime config → same files used while preprocessing.

With those pieces aligned, EveNet can stream data across Ray workers and match the exact tensor schema you curated in the preprocessing config.

