# 🧪 Data Preparation & Updated Input Reference

This page documents the **new preprocessing contract** introduced with the event metadata stored in `share/event_info/pretrain.yaml`. Use it to build the `.npz` files that the EveNet converter ingests and to understand how each tensor maps onto the model heads.

- [Config + CLI workflow](#config--cli-workflow)
- [Input tensor dictionary](#input-tensor-dictionary)
- [Supervision targets by head](#supervision-targets-by-head)
- [NPZ → Parquet conversion](#npz--parquet-conversion)
- [Runtime checklist](#runtime-checklist)

---

<a id="config--cli-workflow"></a>
## 🛠️ Config + CLI Workflow

1. **Start from the event info YAML.** The canonical schema lives in `share/event_info/pretrain.yaml`. The names inside the `INPUTS` block are purely labels used for logging and plotting; what matters is the order, which **must** match the tensor layout you write to disk.
2. **Produce an event dictionary.** Create one Python dictionary per event with the keys and shapes described below. Store the events in a compressed `.npz` file. Masks indicate which padded entries are valid and should contribute to the loss.
3. **Run the EveNet converter.** Point `preprocessing/preprocess.py` at your `.npz` bundle and pass the matching YAML so the loader can recover feature names, the number of sequential vectors, and the heads you are enabling.
4. **Train or evaluate.** Training configs reference the resulting parquet directory via `platform.data_parquet_dir` and reuse the same YAML in `options.Dataset.event_info`.

> ✨ **Normalization note.** The `normalize`, `log_normalize`, and `none` tags in the YAML are metadata only. EveNet derives channel-wise statistics during conversion. The sole special case is `normalize_uniform`, which reserves a transformation for circular variables (`φ`); the model automatically maps to and from the wrapped representation.

---

<a id="input-tensor-dictionary"></a>
## 📦 Input Tensor Dictionary

Each event is described by the following feature tensors. Shapes are shown in `(dim,)` notation for 1‑D arrays and `(rows, cols)` for matrices. Masks share the same leading dimension as the value they gate.

| Key | Shape | Description |
| --- | --- | --- |
| `num_vectors` | `()` | Total count of global + sequential objects. Populate if you have heterogeneous object sets; otherwise leave empty (`[]`) to signal that only sequential objects are present. |
| `num_sequential_vectors` | `()` | Number of valid sequential entries. Mirrors `num_vectors` behaviour. |
| `subprocess_id` | `()` | Integer label identifying the subprocess drawn from the YAML metadata. |
| `x` | `(18, 7)` | Point-cloud tensor with **up to 18 particles** and **7 features** ordered as: energy, `pT`, `η`, `φ`, b‑tag score, lepton flag, charge. Padding is allowed; mark invalid particles with `0` in `x_mask`. |
| `x_mask` | `(18,)` | Boolean (or `0/1`) mask indicating which particle slots in `x` correspond to real objects. Only entries with mask `1` contribute to losses and metrics. |
| `conditions` | `(10,)` | Event-level scalars in the order listed under `INPUTS/GLOBAL/Conditions` in the YAML: `met`, `met_phi`, `nLepton`, `nbJet`, `nJet`, `HT`, `HT_lep`, `M_all`, `M_leps`, `M_bjets`. |
| `conditions_mask` | `(1,)` | Mask for `conditions`. Set to `1` when the global features are present. |

When you feed multiple events, stack each entry along the leading dimension (e.g., `x` becomes `(events, 18, 7)`).

---

<a id="supervision-targets-by-head"></a>
## 🎯 Supervision Targets by Head

Only provide the tensors required for the heads you enable in your training YAML. Omit unused targets or set them to empty arrays so the converter skips unnecessary storage.

### Classification Head

| Key | Shape | Meaning |
| --- | --- | --- |
| `classification` | `()` | Process label per event. Combine with `event_weight` for weighted cross-entropy when sampling imbalanced campaigns. |
| `event_weight` | `()` | Optional sample weight; defaults to `1` if omitted. |

### TruthGeneration Head

| Key | Shape | Meaning |
| --- | --- | --- |
| `x_invisible` | `(N<sub>ν</sub>, 3)` | Invisible particle (e.g., neutrino) features. `N<sub>ν</sub>` is the **maximum** number of invisible objects you intend to pad—`2` in the example. Feature order is defined in your YAML under the TruthGeneration block. |
| `x_invisible_mask` | `(N<sub>ν</sub>,)` | Flags which invisible entries are valid. |
| `num_invisible_raw` | `()` | Count of all invisible objects before quality cuts. |
| `num_invisible_valid` | `()` | Number of invisible objects associated with reconstructed parents. |

### ReconGeneration Head

ReconGeneration is self-supervised: it perturbs the visible point-cloud channels and learns to denoise them. The target specification (which channels to regenerate) lives **directly in the YAML** under the ReconGeneration configuration. No additional tensors beyond the standard inputs are required.

### Resonance Assignment Head

| Key | Shape | Meaning |
| --- | --- | --- |
| `assignments-indices` | `(56, 3)` | Resonance-to-child mapping. Each of the 56 resonances lists up to 3 child indices drawn from the sequential inputs. |
| `assignments-mask` | `(56,)` | Indicates whether **all** children for a given resonance were reconstructed. |
| `assignments-indices-mask` | `(56, 3)` | Per-child validity flags. Use `0` to pad absent daughters while keeping other children active. |

### Segmentation Head

| Key | Shape | Meaning |
| --- | --- | --- |
| `segmentation-class` | `(4, 9)` | One-hot daughter class per slot (4 daughters × 9 resonance types). |
| `segmentation-data` | `(4, 18)` | Assignment of each daughter slot to one of the 18 input particles. |
| `segmentation-momentum` | `(4, 4)` | Ground-truth four-momenta for the segmented daughters. |
| `segmentation-full-class` | `(4, 9)` | Boolean indicator: `1` if all daughters of the resonance are reconstructed. |

If you disable the segmentation head, you can skip all four tensors.

---

<a id="npz--parquet-conversion"></a>
## 🔄 NPZ → Parquet Conversion

1. **Assemble events** into Python lists and save them with `numpy.savez` (or `savez_compressed`). Each key listed above becomes an array inside the archive.
2. **Invoke the converter**:

   ```bash
   python preprocessing/preprocess.py \
     share/event_info/pretrain.yaml \
     --in_npz /path/to/events.npz \
     --store_dir /path/to/output \
     --cpu_max 32
   ```

   The converter reads the YAML to recover feature names, masks, and head activation flags, then emits:

   - `data_*.parquet` containing flattened tensors.
   - `shape_metadata.json` with the original shapes (e.g., `(18, 7)` for `x`).
   - `normalization.pt` with channel-wise statistics and class weights.

3. **Inspect the logs.** The script reports how many particles, invisible objects, and resonances were valid across the dataset—helpful when debugging mask alignment.

---

<a id="runtime-checklist"></a>
## ✅ Runtime Checklist

- `platform.data_parquet_dir` points to the folder with the generated parquet shards and `shape_metadata.json`.
- `options.Dataset.event_info` references the same YAML (`share/event_info/pretrain.yaml` or your copy).
- `options.Dataset.normalization_file` matches the `normalization.pt` produced during conversion.
- Only the heads you activated in the training YAML have matching supervision tensors in the parquet files.

With those pieces in place, EveNet will rebuild the full event dictionary on the fly, apply the appropriate circular normalization for `normalize_uniform` channels, and route each tensor to the corresponding head.

