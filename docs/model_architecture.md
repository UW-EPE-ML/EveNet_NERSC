# 🧠 Model Architecture Tour

Take a guided walk through EveNet’s multitask architecture—from input normalization to the specialized heads. Pair this with the [configuration guide](configuration.md) to see how YAML choices shape each component.

- [Signal flow at a glance](#signal-flow)
- [Input normalization](#input-normalization)
- [Shared body](#shared-body)
- [Task heads](#task-heads)
- [Progressive training hooks](#progressive-training)
- [Customizing the network](#customizing)

---

## 🔁 Signal Flow at a Glance {#signal-flow}

```
Inputs → Normalizers → Global Embedding ┐
                               PET Body ├─> Object Encoder ─┬─ Classification
                                                            ├─ Regression
                                                            ├─ Assignment
                                                            ├─ Segmentation
                                                            ├─ Point-Cloud Generation
                                                            └─ Neutrino Generation
```

Every stage is instantiated inside [`evenet/network/evenet_model.py`](../evenet/network/evenet_model.py) using the options loaded from your YAML files.

---

## 🧴 Input Normalization {#input-normalization}

When `EveNetModel` is built, it grabs feature statistics from `normalization.pt` plus schema details from `event_info` and constructs a collection of `Normalizer` layers:

- **Sequential features** (`INPUTS/Source`) use a `Normalizer` that understands mixed discrete/continuous distributions and optional inverse-CDF indices.
- **Global features** (`INPUTS/Conditions`) map through a second `Normalizer` sized to the event-level vector.
- **Multiplicity channels** (`num_vectors`, `num_sequential_vectors`) are normalized when generation heads are active.
- **Invisible particles** share embedding widths with sequential features and are padded to `max_neutrinos` so diffusion heads can operate consistently.

Implementation details live near the top of [`evenet/network/evenet_model.py`](../evenet/network/evenet_model.py#L1-L120).

---

## 🧱 Shared Body {#shared-body}

### 🌐 Global Embedding
`GlobalVectorEmbedding` converts the condition vector into learned tokens. Hyperparameters like depth, hidden dimension, dropout, and activation come from `Body.GlobalEmbedding` in your network YAML (e.g., [`share/network/network-20M.yaml`](../share/network/network-20M.yaml#L1-L18)).

### 🧲 PET Body
`PETBody` processes the sequential particle cloud with transformer-style layers, local neighborhood attention, and optional stochastic feature dropping. Configure `num_layers`, `num_heads`, `feature_drop`, and `local_point_index` under `Body.PET` (see [`network-20M.yaml`](../share/network/network-20M.yaml#L14-L32)).

### 🧵 Object Encoder
Outputs from the PET body and global tokens meet in the `ObjectEncoder`, which mixes information across objects. Attention depth, head counts, positional embedding size, and skip connections are controlled by `Body.ObjectEncoder` (see [`network-20M.yaml`](../share/network/network-20M.yaml#L33-L46)).

---

## 🎯 Task Heads {#task-heads}

Heads are instantiated only when `options.Training.Components.<Head>.include` is `true`.

### 🏷️ Classification
Predicts process probabilities using `ClassificationHead`. Configure layer counts, hidden size, dropout, and optional attention under `Classification` in the network YAML ([`network-20M.yaml`](../share/network/network-20M.yaml#L48-L60)).

### 📈 Regression
`RegressionHead` regresses continuous targets (momenta, masses). Normalization tensors (`regression_mean`, `regression_std`) are injected so outputs can be de-standardized. Hyperparameters mirror the classification head ([`network-20M.yaml`](../share/network/network-20M.yaml#L62-L70)).

### 🔗 Assignment
`SharedAssignmentHead` tackles combinatorial matching between reconstructed objects and truth daughters defined in `event_info`. It leverages symmetry-aware attention and optional detection layers. Tune `feature_drop`, attention heads, and decoder depth via the `Assignment` block ([`network-20M.yaml`](../share/network/network-20M.yaml#L72-L108)).

### 🌈 Segmentation
`SegmentationHead` predicts binary masks for resonance-specific particle groups. Configure the number of queries, transformer layers, and projection widths in the `Segmentation` block ([`network-20M.yaml`](../share/network/network-20M.yaml#L108-L126)).

### 🌬️ Generation Family
Three diffusion-based heads share the `EventGenerationHead` implementation:

| Head | Purpose | Key knobs |
| --- | --- | --- |
| `GlobalGeneration` | Diffuse event-level scalars (multiplicities, global targets). | `diffusion_steps`, latent dimensions, conditioning indices ([`network-20M.yaml`](../share/network/network-20M.yaml#L110-L126)). |
| `ReconGeneration` | Reconstruct particle-cloud features conditioned on PET embeddings. | Layer scale, dropout probability, hidden width ([`network-20M.yaml`](../share/network/network-20M.yaml#L128-L146)). |
| `TruthGeneration` | Generate invisible particles (e.g., neutrinos). | Shares configuration with reconstruction but adapts output dimensionality ([`network-20M.yaml`](../share/network/network-20M.yaml#L128-L146)). |

---

## 🌀 Progressive Training Hooks {#progressive-training}

`EveNetModel` exposes `schedule_flags` describing which heads are active (diffusion, neutrino, deterministic). The training loop combines these flags with the curriculum defined in `options.ProgressiveTraining` so that loss weights, dropout, or EMA decay ramp smoothly over time. Inspect the scheduling logic in [`evenet/network/evenet_model.py`](../evenet/network/evenet_model.py#L352-L380) and pair it with the YAML stages in [`share/options/options.yaml`](../share/options/options.yaml#L160-L218).

---

## 🛠️ Customizing the Network {#customizing}

1. **Pick a template** – start from `share/network/network-20M.yaml` or another file under `share/network/`.
2. **Override selectively** – in your top-level YAML, override only the fields you want to tweak (e.g., set `Body.PET.feature_drop: 0.0` for fine-tuning).
3. **Match supervision** – enable heads under `options.Training.Components` only when the dataset provides the required targets.
4. **Refresh normalization** – if you change the input schema in `event_info`, rerun preprocessing so new statistics land in `normalization.pt` (see the saving logic in [`preprocessing/postprocessor.py`](../preprocessing/postprocessor.py#L360-L406)).

With these controls, you can resize EveNet for tiny studies or scale it up for production campaigns—all while keeping the codepath consistent.

