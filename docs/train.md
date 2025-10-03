# Training

> 🚀 **Quickstart:** Install the package with `pip install evenet` and run `evenet-train <config.yaml>`. Developing from source? Use `python -m evenet.train <config.yaml>` to pick up local edits.

## Loading and Saving Models

This guide outlines how model weights and Exponential Moving Average (EMA) weights are handled in the training workflow.
It supports both standard training continuation and pretraining-based initialization.

---

### 📦 YAML Configuration

Specify the following fields under `options.Training` in your YAML config:

```yaml
model_checkpoint_save_path: "."                  # Directory to save Lightning checkpoints
model_checkpoint_load_path: null                 # Path to resume training from a checkpoint (.ckpt)
pretrain_model_load_path: null                   # Path to load pretrained model weights

EMA:
  enable: true                                   # Enable Exponential Moving Average tracking
  decay: 0.999                                   # Decay rate for EMA updates
  replace_model_after_load: false                # Use EMA weights to overwrite model after load
  replace_model_at_end: true                     # Use EMA weights to overwrite model before saving
```

---

### 🔁 Resuming Training from Checkpoint

When `model_checkpoint_load_path` is provided, PyTorch Lightning automatically:

* Restores the model weights from `checkpoint["state_dict"]`
* Resumes the optimizer, scheduler, and training state (e.g., `global_step`, `current_epoch`)

If `EMA.enable: true`, the training script additionally:

* Loads EMA weights from `checkpoint["ema_state_dict"]`
* Optionally replaces the main model weights with EMA if `EMA.replace_model_after_load: true`

---

### 🚀 Initializing from Pretrained Model

When `pretrain_model_load_path` is specified, the system loads model weights during `configure_model()` using
shape-validated safe loading:

* Only layers with matching names and shapes are loaded
* Incompatible layers are skipped with informative warnings

This is suitable for transfer learning or domain adaptation tasks.

> **Note on EMA:**
>
> * EMA weights are not loaded from the pretrained model
> * If `EMA.enable: true`, the EMA model is initialized from the current model after loading

---

### 📂 Saving Checkpoints

When saving a checkpoint (e.g., at the end of training), Lightning includes:

* Model state dict
* Optimizer and scheduler state
* Training progress (epoch, global step, etc.)

If `EMA.replace_model_at_end: true`, the system first copies EMA weights into the model before saving. This ensures the
checkpoint reflects the EMA-smoothed model.

---

### ✅ Summary of Loading and Saving Behavior

| Scenario                   | YAML Setting                 | Main Model Loaded | EMA Loaded        | EMA Replaces Model                  |
|----------------------------|------------------------------|-------------------|-------------------|-------------------------------------|
| Resume from checkpoint     | `model_checkpoint_load_path` | ✅ (automatic)     | ✅ if `EMA.enable` | ✅ if `EMA.replace_model_after_load` |
| Load from pretrained model | `pretrain_model_load_path`   | ✅ (safe-load)     | ❌                 | ✅ if `EMA.replace_model_after_load` |
| Save at end of training    | `model_checkpoint_save_path` | ✅                 | ✅ if `EMA.enable` | ✅ if `EMA.replace_model_at_end`     |
