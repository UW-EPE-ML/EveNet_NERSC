# 🧠 Model Architecture Tour

Take a guided walk through EveNet’s multitask architecture—from input normalization to the specialized heads. Pair this with the [configuration guide](configuration.md) to see how YAML choices shape each component.

- [Signal flow at a glance](#signal-flow)
- [Input normalization](#input-normalization)
- [Shared body](#shared-body)
- [Task heads](#task-heads)
- [Progressive training hooks](#progressive-training)
- [Customizing the network](#customizing)

---

<a id="signal-flow"></a>
## 🔁 Signal Flow at a Glance

```mermaid
flowchart LR
    subgraph Inputs
        A[(Point Cloud)]
        B[(Global Conditions)]
        C[(Invisible Particles)]
    end
    subgraph Normalizers
        N1[Sequential
        Normalizer]
        N2[Global
        Normalizer]
        N3[Invisible
        Normalizer]
    end
    subgraph SharedBody
        GE[Global Embedding]
        PET[PET Body]
        OE[Object Encoder]
    end
    subgraph DiscriminativeHeads
        CLF[Classification]
        REG[Regression]
        ASN[Assignment]
        SEG[Segmentation]
    end
    subgraph GenerativeHeads
        GG[Global Generation
        (separate network)]
        RG[Recon Generation]
        TG[Truth Generation]
    end
    A --> N1 --> PET
    B --> N2 --> GE
    C --> N3
    N1 -->|mask align| PET
    N3 --> PET
    GE --> PET
    GE --> GG
    PET --> OE
    OE --> CLF
    OE --> REG
    OE --> ASN
    OE --> SEG
    PET --> RG
    PET --> TG
    GE --> RG
    GE --> TG
```

Every stage is instantiated inside [`evenet/network/evenet_model.py`](../evenet/network/evenet_model.py) using the options loaded from your YAML files.

---

<a id="input-normalization"></a>
## 🧴 Input Normalization

When `EveNetModel` is built, it grabs feature statistics from `normalization.pt` plus schema details from `event_info` and constructs a collection of `Normalizer` layers:

- **Sequential features** (`INPUTS/Source`) use a `Normalizer` that understands mixed discrete/continuous distributions and optional inverse-CDF indices.
- **Global features** (`INPUTS/Conditions`) map through a second `Normalizer` sized to the event-level vector.
- **Multiplicity channels** (`num_vectors`, `num_sequential_vectors`) are normalized when generation heads are active.
- **Invisible particles** share embedding widths with sequential features and are padded to `max_neutrinos` so diffusion heads can operate consistently.

Implementation details live near the top of [`evenet/network/evenet_model.py`](../evenet/network/evenet_model.py#L1-L120).

---

<a id="shared-body"></a>
## 🧱 Shared Body

### 🌐 Global Embedding
`GlobalVectorEmbedding` converts the condition vector into learned tokens. Hyperparameters like depth, hidden dimension, dropout, and activation come from the `Body.GlobalEmbedding` block described in the [configuration reference](configuration.md#network-templates).

### 🧲 PET Body
`PETBody` processes the sequential particle cloud with transformer-style layers, local neighborhood attention, and optional stochastic feature dropping. Configure `num_layers`, `num_heads`, `feature_drop`, and `local_point_index` under `Body.PET` in your network block (see [configuration reference](configuration.md#network-templates)).

### 🧵 Object Encoder
Outputs from the PET body and global tokens meet in the `ObjectEncoder`, which mixes information across objects. Attention depth, head counts, positional embedding size, and skip connections are controlled by `Body.ObjectEncoder` (see [configuration reference](configuration.md#network-templates)).

---

<a id="task-heads"></a>
## 🎯 Task Heads

Heads are instantiated only when `options.Training.Components.<Head>.include` is `true`.

### 🏷️ Classification
Predicts process probabilities using `ClassificationHead`. Configure layer counts, hidden size, dropout, and optional attention under `Classification` in the network YAML (see [configuration reference](configuration.md#network-templates)).

### 📈 Regression
`RegressionHead` regresses continuous targets (momenta, masses). Normalization tensors (`regression_mean`, `regression_std`) are injected so outputs can be de-standardized. Hyperparameters mirror the classification head (see [configuration reference](configuration.md#network-templates)).

### 🔗 Assignment
`SharedAssignmentHead` tackles combinatorial matching between reconstructed objects and truth daughters defined in `event_info`. It leverages symmetry-aware attention and optional detection layers. Tune `feature_drop`, attention heads, and decoder depth via the `Assignment` block (see [configuration reference](configuration.md#network-templates)).

### 🌈 Segmentation

`SegmentationHead` predicts binary masks for resonance-specific particle groups. Configure the number of queries, transformer layers, and projection widths in the `Segmentation` block (see [configuration reference](configuration.md#network-templates)).

### 🌬️ Generation Family
EveNet carries **three** diffusion-based heads, all orchestrated in the forward pass but connected differently:

| Head | Input features | Output target | Notes |
| --- | --- | --- | --- |
| `GlobalGeneration` | Only the normalized global tokens and multiplicities (no PET/object encoder). | Event-level scalars such as multiplicity counts. | Implemented as an independent network; think of it as a standalone diffusion chain that shares only the label embedding. Hyperparameters live under `GlobalGeneration` in the network block. |
| `ReconGeneration` | PET embeddings + global tokens before the object encoder. | Visible point-cloud features (`INPUTS/Source`). | Shares the PET backbone directly—noise is injected on the sequential features and denoised with the same PET parameters. Configure under `ReconGeneration`. |
| `TruthGeneration` | PET embeddings + global tokens with optional invisible padding. | Invisible particle features (e.g., neutrinos). | Uses the same architecture family as reconstruction but targets the padded invisible channels. Settings sit under `TruthGeneration`. |

---

<a id="progressive-training"></a>
## 🌀 Progressive Training Hooks

`EveNetModel` exposes `schedule_flags` describing which heads are active (diffusion, neutrino, deterministic). The training loop combines these flags with the curriculum defined in `options.ProgressiveTraining` so that loss weights, dropout, or EMA decay ramp smoothly over time. Inspect the scheduling logic in [`evenet/network/evenet_model.py`](../evenet/network/evenet_model.py#L352-L380) and pair it with the YAML stages summarized in the [configuration reference](configuration.md#options-deep-dive).

---

<a id="customizing"></a>
## 🛠️ Customizing the Network

1. **Pick a template** – choose a network block described in the [configuration reference](configuration.md#network-templates) and copy it into your experiment YAML.
2. **Override selectively** – in your top-level YAML, override only the fields you want to tweak (e.g., set `Body.PET.feature_drop: 0.0` for fine-tuning).
3. **Match supervision** – enable heads under `options.Training.Components` only when the dataset provides the required targets.
4. **Refresh normalization** – if you change the input schema in `event_info`, rerun preprocessing so new statistics land in `normalization.pt` (see the saving logic in [`preprocessing/postprocessor.py`](../preprocessing/postprocessor.py#L360-L406)).

With these controls, you can resize EveNet for tiny studies or scale it up for production campaigns—all while keeping the codepath consistent.

