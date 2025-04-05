import json
import math
import pickle

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import torch
from evenet.control.global_config import global_config
from evenet.dataset.preprocess import process_event_batch, convert_batch_to_torch_tensor
from evenet.network.evenet_model import EvenetModel

from evenet.network.loss.classification import loss as cls_loss
from evenet.network.loss.regression import loss as reg_loss

from preprocessing.preprocess import unflatten_dict

import torch
import torch.nn as nn
from collections import defaultdict


class DebugHookManager:
    def __init__(self, track_forward=True, track_backward=True, save_values=False):
        self.forward_hooks = []
        self.backward_hooks = []
        self.grad_hooks = []
        self.save_values = save_values

        self.forward_outputs = defaultdict(list)
        self.backward_grads = defaultdict(list)

        self.track_forward = track_forward
        self.track_backward = track_backward

    def check_forward(self, name):
        def hook(module, input, output):
            # Check output
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    print(f"[NaN Detected] Forward output of {name}")
                if torch.isinf(output).any():
                    print(f"[Inf Detected] Forward output of {name}")
                if self.save_values:
                    self.forward_outputs[name].append(output.detach().cpu())
            # Check inputs
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    if torch.isnan(inp).any():
                        print(f"[NaN Detected] Forward input {i} of {name}")
                    if torch.isinf(inp).any():
                        print(f"[Inf Detected] Forward input {i} of {name}")

        return hook

    def check_backward(self, name):
        def hook(module, grad_input, grad_output):
            for i, g in enumerate(grad_input):
                if isinstance(g, torch.Tensor):
                    if torch.isnan(g).any():
                        print(f"[NaN Detected] Grad input {i} of {name}")
                    if torch.isinf(g).any():
                        print(f"[Inf Detected] Grad input {i} of {name}")
                    if self.save_values:
                        self.backward_grads[name].append(g.detach().cpu())

        return hook

    def check_param_grad(self, name, param):
        def hook(grad):
            if grad is not None:
                if torch.isnan(grad).any():
                    print(f"[NaN Detected] Grad of param {name}")
                if torch.isinf(grad).any():
                    print(f"[Inf Detected] Grad of param {name}")
                if self.save_values:
                    self.backward_grads[f"param::{name}"].append(grad.detach().cpu())

        return hook

    def attach_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, nn.Module) and len(list(module.children())) == 0:  # only leaf modules
                if self.track_forward:
                    fh = module.register_forward_hook(self.check_forward(name))
                    self.forward_hooks.append(fh)
                if self.track_backward:
                    bh = module.register_full_backward_hook(self.check_backward(name))
                    self.backward_hooks.append(bh)

        for name, param in model.named_parameters():
            if param.requires_grad:
                gh = param.register_hook(self.check_param_grad(name, param))
                self.grad_hooks.append(gh)

    def remove_hooks(self):
        for h in self.forward_hooks + self.backward_hooks:
            h.remove()
        self.forward_hooks.clear()
        self.backward_hooks.clear()
        self.grad_hooks.clear()
        print("✅ All hooks removed.")

    def dump_debug_data(self):
        # Optional utility: Save collected outputs/grads to disk or analyze
        print("🔍 Dumped forward activations:")
        for k, v in self.forward_outputs.items():
            print(f"{k}: {len(v)} tensors")

        print("🔍 Dumped backward gradients:")
        for k, v in self.backward_grads.items():
            print(f"{k}: {len(v)} tensors")


########################
## Debug configuation ##
########################

wandb_enable = True
n_epoch = 1
df_number = 5000
num_splits = 10
debugger_enable = True

global_config.load_yaml("/Users/avencastmini/PycharmProjects/EveNet/share/local_test.yaml")

shape_metadata = json.load(
    open("/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output/shape_metadata.json"))

# Load the Parquet file locally
df = pq.read_table(
    "/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output/data_run_yulei_11.parquet").to_pandas()

# Optional: Subsample for speed
df = df.head(df_number)

# Convert to dict-of-arrays if needed
batch = {col: df[col].to_numpy() for col in df.columns}

# Preprocess batch
processed_batch = process_event_batch(batch, shape_metadata=shape_metadata, unflatten=unflatten_dict)

# Convert to torch
torch_batch = convert_batch_to_torch_tensor(processed_batch)

# with open("/Users/avencastmini/PycharmProjects/EveNet/workspace/normalization_file/PreTrain_norm.pkl", 'rb') as f:
#     normalization_file = pickle.load(f)


# Run forward
model = EvenetModel(
    config=global_config,
    device=torch.device("cpu"),
)
model.train()


debugger = DebugHookManager(track_forward=True, track_backward=True, save_values=True)
if debugger_enable:
    debugger.attach_hooks(model)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch_size = len(df)
split_batches = []

for i in range(num_splits):
    start = math.floor(i * batch_size / num_splits)
    end = math.floor((i + 1) * batch_size / num_splits)
    split = {k: v[start:end] for k, v in torch_batch.items()}
    split_batches.append(split)

if wandb_enable:
    import wandb

    wandb.init(
        project="debug",
        entity=global_config.wandb.entity
    )

for iepoch in range(n_epoch):
    for i, batch in enumerate(split_batches):
        with torch.autograd.set_detect_anomaly(True):
            for name, tensor in batch.items():
                if torch.isnan(tensor).any():
                    print(f"[Batch {i}] NaN found in input tensor: {name}")

            outputs = model.shared_step(batch, batch_size=len(batch["classification"]))

            # Regression
            reg_output = outputs["regression"]
            flattened = torch.cat([v.squeeze(0) for v in reg_output.values()], dim=-1)
            if torch.isnan(flattened).any():
                print(f"[Batch {i}] NaN in regression output")

            reg_target = batch["regression-data"].float()
            reg_mask = batch["regression-mask"].float()

            r_loss = reg_loss(predict=flattened, target=reg_target, mask=reg_mask)
            if torch.isnan(r_loss).any():
                print(f"[Batch {i}] NaN in regression loss")

            # Classification
            cls_output = next(iter(outputs["classification"].values()))
            cls_target = batch["classification"]

            c_loss = cls_loss(predict=cls_output, target=cls_target)
            if torch.isnan(c_loss).any():
                print(f"[Batch {i}] NaN in classification loss")

            total_loss = c_loss.mean()  # + r_loss.mean() * 0.1

            print(f"[Batch {i}] Total loss: {total_loss}", flush=True)

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"[Batch {i}] Done")

            if wandb_enable:
                wandb.log(
                    {
                        "epoch": iepoch,
                        "total_loss": total_loss.item(),
                        "classification_loss": c_loss.mean().item(),
                        "regression_loss": r_loss.mean().item()}
                )
        # break


if debugger_enable:
    debugger.remove_hooks()
    debugger.dump_debug_data()