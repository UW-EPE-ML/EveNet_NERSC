# Model Architecture Guide

EveNet assembles a multitask architecture that normalizes heterogeneous physics inputs, processes them with shared transformers, and branches into task-specific heads. This guide explains each stage and how configuration options shape the network.

```
Inputs → Normalizers → Global Embedding ┐
                               PET Body ├─> Object Encoder ─┬─ Classification
                                                            ├─ Regression
                                                            ├─ Assignment
                                                            ├─ Segmentation
                                                            ├─ Point-Cloud Generation
                                                            └─ Neutrino Generation
```

## Input Normalization

When `EveNetModel` is constructed it collects normalization statistics from `normalization.pt` and the feature schema defined in `event_info`. Separate `Normalizer` layers handle sequential particles, global conditions, optional point-cloud counts, and invisible particles. Each normalizer receives mean/std tensors derived during preprocessing and boolean masks that indicate which features should be normalized.【F:evenet/network/evenet_model.py†L1-L120】

- **Sequential features** (`INPUTS/Source`) use `Normalizer` with optional inverse-CDF indices for mixed discrete/continuous distributions.
- **Global features** (`INPUTS/Conditions`) map to a second `Normalizer` sized by the event-level feature count.
- **Point-cloud multiplicities** (`input_num_*`) are normalized when the point-cloud generation head is enabled.
- **Invisible particles** reuse the sequential feature dimension with optional zero-padding so diffusion heads can operate on the same embedding width.【F:evenet/network/evenet_model.py†L41-L118】

## Shared Body

### Global Embedding

`GlobalVectorEmbedding` transforms the event-level condition vector into a learned embedding. Hyperparameters such as embedding depth, hidden dimension, dropout, and activation are set under `Body.GlobalEmbedding` in the network YAML (e.g., GRU blocks with 256 hidden units in `network-20M.yaml`).【F:evenet/network/evenet_model.py†L120-L160】【F:share/network/network-20M.yaml†L1-L32】

### PET Body

`PETBody` ingests the sequential particle cloud using transformer layers, optional local neighborhood attention, and stochastic feature dropping. Configuration fields like `num_layers`, `num_heads`, `feature_drop`, and `local_point_index` determine receptive field and regularization strength.【F:evenet/network/evenet_model.py†L120-L160】【F:share/network/network-20M.yaml†L14-L32】

### Object Encoder

Outputs from the PET body feed into an `ObjectEncoder` that mixes particle and global tokens. The encoder’s attention heads, depth, positional embedding size, and skip connections are controlled by `Body.ObjectEncoder` in the network YAML.【F:evenet/network/evenet_model.py†L160-L176】【F:share/network/network-20M.yaml†L33-L46】

## Task Heads

Each task is optional and activated by the `options.Training.Components.<Task>.include` flags. When enabled, EveNet instantiates the corresponding head with parameters pulled from the network YAML.

### Classification

Predicts process probabilities using `ClassificationHead`. It consumes the encoded objects, applies multi-head attention if requested, and outputs logits for every process defined in `event_info`. The number of layers, hidden size, dropout, and skip connections come from `Classification` in the network config.【F:evenet/network/evenet_model.py†L176-L199】【F:share/network/network-20M.yaml†L48-L60】

### Regression

`RegressionHead` regresses momenta and other continuous targets. It receives normalization tensors (`regression_mean`/`regression_std`) to de-standardize outputs during inference. Configuration mirrors the classification head but without attention-specific knobs.【F:evenet/network/evenet_model.py†L199-L214】【F:share/network/network-20M.yaml†L62-L70】

### Assignment

`SharedAssignmentHead` solves combinatorial assignments between reconstructed objects and truth daughters defined in `event_info.event_particles`. It leverages symmetry-aware attention and optional detection layers. The head relies on process-specific pairing topology, permutation catalogs, and particle symmetries also loaded from `event_info`. Tunable options (feature drop, attention heads, detection depth) live in the `Assignment` block of the network YAML.【F:evenet/network/evenet_model.py†L214-L262】【F:share/network/network-20M.yaml†L72-L108】

### Generation Heads

- **GlobalConditional Generation** (`GlobalGeneration`) diffuses over event-level scalar targets (e.g., multiplicities). Inputs include both the diffusion time and condition indices configured in `event_info` and the YAML. Layer counts, hidden sizes, and resnet dimensions are configurable.【F:evenet/network/evenet_model.py†L262-L290】【F:share/network/network-20M.yaml†L110-L126】
- **Reconstruction Generation** (`ReconGeneration`) produces point-cloud features conditioned on the PET embeddings and class labels. Hyperparameters mirror the PET body but with diffusion-specific knobs like `layer_scale` and `drop_probability`.【F:evenet/network/evenet_model.py†L290-L311】【F:share/network/network-20M.yaml†L128-L146】
- **Truth Generation** (`TruthGeneration`) models invisible particles (e.g., neutrinos). It shares the `EventGenerationHead` implementation but adapts output dimensionality and can optionally encode positional information for variable-length targets.【F:evenet/network/evenet_model.py†L311-L331】【F:share/network/network-20M.yaml†L128-L146】

### Segmentation

`SegmentationHead` predicts binary masks for resonance-specific particle groups. The number of queries and transformer layers determines how many candidate segments are produced per event. Configuration fields include projection dimension, number of heads, and whether to return intermediate decoder layers.【F:evenet/network/evenet_model.py†L331-L352】【F:share/network/network-20M.yaml†L108-L126】

## Progressive Training & Scheduling

`EveNetModel` exposes `schedule_flags` that describe which heads are active (generation, neutrino generation, deterministic tasks). The training engine combines these flags with the progressive training schedule defined in `options/options.yaml` to ramp losses, dropout, and EMA decay across curriculum stages.【F:evenet/network/evenet_model.py†L352-L380】【F:share/options/options.yaml†L160-L218】

During `forward`, the model expects the tensors produced by preprocessing (point cloud, masks, assignments, regressions, segmentation, invisible particles). These inputs are normalized, embedded, and dispatched to each active head, producing a dictionary of task outputs ready for Lightning loss computation.【F:evenet/network/evenet_model.py†L1-L120】【F:evenet/network/evenet_model.py†L352-L380】

## Customizing the Architecture

1. Choose a base template under `share/network/` (`network-20M.yaml` for the compact model, `network-100M.yaml` for a larger footprint).
2. Override specific fields in your top-level config’s `network` section (e.g., set `Body.PET.feature_drop: 0.0` to disable stochastic feature dropping during fine-tuning).【F:share/finetune-example.yaml†L120-L133】
3. Enable or disable task heads via `options.Training.Components`. Only instantiate heads you have supervision for to save memory and compute.【F:share/finetune-example.yaml†L60-L107】
4. Regenerate normalization statistics if you change the input schema in `event_info` so the normalizers remain consistent.【F:preprocessing/postprocessor.py†L360-L406】

By aligning the network YAML, options file, and preprocessing schema, you can tailor EveNet’s capacity to new physics targets while keeping the shared training loop intact.
