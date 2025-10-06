# 🧪 Data Preparation & Updated Input Reference

This page documents the **new preprocessing contract** for EveNet. The reference schema in `share/event_info/pretrain.yaml` is bundled with the repository as an example, but you are free to adapt the feature lists to match your campaign. The goal of this guide is to help you build the `.npz` files that the EveNet converter ingests and to clarify how each tensor maps onto the model heads.

- [Config + CLI workflow](#config--cli-workflow)
- [Input tensor dictionary](#input-tensor-dictionary)
- [Supervision targets by head](#supervision-targets-by-head)
- [NPZ → Parquet conversion](#npz--parquet-conversion)
- [Runtime checklist](#runtime-checklist)

---

<a id="config--cli-workflow"></a>
## 🛠️ Config + CLI Workflow

1. **Start from an event-info YAML.** The repository ships an example at `share/event_info/pretrain.yaml`; copy or extend it to describe the objects, global variables, and heads you plan to enable. The display names inside the `INPUTS` block are just labels used for logging and plotting—what matters is the order, which **must** match the tensor layout you write to disk.
2. **Produce an event dictionary.** Create one Python dictionary per event with the keys and shapes described below. Store the events in a compressed `.npz` file. Masks indicate which padded entries are valid and should contribute to the loss.
3. **Run the EveNet converter.** Point `preprocessing/preprocess.py` at your `.npz` bundle and pass the matching YAML so the loader can recover feature names, the number of sequential vectors, and the heads you are enabling.
4. **Train or evaluate.** Training configs reference the resulting parquet directory via `platform.data_parquet_dir` and reuse the same YAML in `options.Dataset.event_info`.

> ✨ **Normalization note.** The `normalize`, `log_normalize`, and `none` tags in the YAML are metadata only. EveNet derives channel-wise statistics during conversion. The sole special case is `normalize_uniform`, which reserves a transformation for circular variables (`φ`); the model automatically maps to and from the wrapped representation.

---

<a id="input-tensor-dictionary"></a>
## 📦 Input Tensor Dictionary

Each event is described by the following feature tensors. Shapes are shown with a leading `N` to indicate the number of stored events in a given `.npz` file. Masks share the same leading dimension as the value they gate.

| Key | Shape | Description |
| --- | --- | --- |
| `num_vectors` | `(N,)` | Total count of global + sequential objects per event. Populate if you have heterogeneous object sets; otherwise leave empty (`[]`) to signal that only sequential objects are present. |
| `num_sequential_vectors` | `(N,)` | Number of valid sequential entries per event. Mirrors `num_vectors` behaviour. |
| `subprocess_id` | `(N,)` | Integer label identifying the subprocess drawn from the YAML metadata. |
| `x` | `(N, P, F)` | Point-cloud tensor with **up to** `P` particles and `F` features. In the example YAML, `P = 18` and `F = 7` ordered as energy, `pT`, `η`, `φ`, b‑tag score, lepton flag, charge. Padding is allowed; mark invalid particles with `0` in `x_mask`. |
| `x_mask` | `(N, P)` | Boolean (or `0/1`) mask indicating which particle slots in `x` correspond to real objects. Only entries with mask `1` contribute to losses and metrics. |
| `conditions` | `(N, C)` | Event-level scalars. `C` is the number of global variables you define (10 in the example). You may add or remove variables as long as the order matches your YAML. |
| `conditions_mask` | `(N, 1)` | Mask for `conditions`. Set to `1` when the global features are present. |

> 💡 **Tip:** If you only have a single event while debugging, `N = 1`. The EveNet converter flattens the leading dimension when it writes parquet shards, so you can use any event count per `.npz` bundle.

---

<a id="supervision-targets-by-head"></a>
## 🎯 Supervision Targets by Head

Only provide the tensors required for the heads you enable in your training YAML. Omit unused targets or set them to empty arrays so the converter skips unnecessary storage.

### Classification Head

| Key | Shape | Meaning |
| --- | --- | --- |
| `classification` | `(N,)` | Process label per event. Combine with `event_weight` for weighted cross-entropy when sampling imbalanced campaigns. |
| `event_weight` | `(N,)` | Optional sample weight; defaults to `1` if omitted. |

### TruthGeneration Head

| Key | Shape | Meaning |
| --- | --- | --- |
| `x_invisible` | `(N, N_\nu, F_\nu)` | Invisible particle (e.g., neutrino) features. `N_\nu` is the **maximum** number of invisible objects you intend to pad—`2` in the example—and `F_\nu` is the number of features per invisible. Feature order is defined in your YAML under the TruthGeneration block. |
| `x_invisible_mask` | `(N, N_\nu)` | Flags which invisible entries are valid. |
| `num_invisible_raw` | `(N,)` | Count of all invisible objects before quality cuts. |
| `num_invisible_valid` | `(N,)` | Number of invisible objects associated with reconstructed parents. |

### ReconGeneration Head

ReconGeneration is self-supervised: it perturbs the visible point-cloud channels and learns to denoise them. The target specification (which channels to regenerate) lives **directly in the YAML** under the ReconGeneration configuration. No additional tensors beyond the standard inputs are required.

### Resonance Assignment Head

| Key | Shape | Meaning |
| --- | --- | --- |
| `assignments-indices` | `(N, R, D)` | Resonance-to-child mapping. `R` is the number of resonances you describe in the YAML (56 in the example) and `D` is the maximum number of children a resonance may have (3 in the example). Each entry stores indices drawn from the sequential inputs. |
| `assignments-mask` | `(N, R)` | Indicates whether **all** children for a given resonance were reconstructed. |
| `assignments-indices-mask` | `(N, R, D)` | Per-child validity flags. Use `0` to pad absent daughters while keeping other children active. |

> 📐 **Assignment internals:** During conversion EveNet scans your assignment map to determine `R` and `D`, initialises arrays filled with `-1`, and then writes the actual child indices along with boolean masks. The snippet below mirrors the loader logic so you can generate matching tensors in your own pipeline:

```python
full_indices = np.full((num_events, n_targets, max_daughters), -1, dtype=int)
full_mask = np.zeros((num_events, n_targets), dtype=bool)
index_mask = np.zeros((num_events, n_targets, max_daughters), dtype=bool)
# Fill with your resonance→daughter mappings; mark valid entries in the masks.
```

### Segmentation Head

| Key | Shape | Meaning |
| --- | --- | --- |
| `segmentation-class` | `(N, S, T)` | One-hot daughter class per slot. `S` is the maximum number of segments per event (4 in the example) and `T` is the number of resonance tags (9 in the example). |
| `segmentation-data` | `(N, S, P)` | Assignment of each daughter slot to one of the `P` input particles used in the point cloud. |
| `segmentation-momentum` | `(N, S, 4)` | Ground-truth four-momenta for the segmented daughters. |
| `segmentation-full-class` | `(N, S, T)` | Boolean indicator: `1` if all daughters of the resonance are reconstructed. |

As with assignments, the converter computes `S` and `T` by scanning your segment-tag map. The internal helper creates zero-initialised arrays sized `(num_events, max_event_segments, max_segment_tags)` and fills them according to the resonance definitions. Any slots you leave unused remain associated with the catch-all background class (`0`).

If you disable the segmentation head, you can skip all four tensors.

### Worked Input Example

```python
import numpy as np

example = {
    "x": np.zeros((N, 18, 7), dtype=np.float32),
    "x_mask": np.zeros((N, 18), dtype=bool),
    "conditions": np.zeros((N, 12), dtype=np.float32),  # customise global count
    "conditions_mask": np.ones((N, 1), dtype=bool),
    "assignments-indices": np.full((N, R, D), -1, dtype=int),
    "assignments-mask": np.zeros((N, R), dtype=bool),
    "segmentation-data": np.zeros((N, S, 18), dtype=bool),
    # ... add heads you enabled ...
}
```

Feel free to increase or shrink the dimensions (`P`, `C`, `R`, `D`, `S`, `T`) to match your physics process. Just keep the YAML in sync so the converter knows how many channels to expect.

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

