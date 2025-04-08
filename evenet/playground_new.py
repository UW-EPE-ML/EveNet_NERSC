import json
import math
import pickle

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import torch
from evenet.control.global_config import global_config
from evenet.dataset.preprocess import process_event_batch, convert_batch_to_torch_tensor
from evenet.network.evenet_model import EveNetModel

from evenet.network.loss.classification import loss as cls_loss
from evenet.network.loss.regression import loss as reg_loss
from evenet.network.loss.assignment import loss as ass_loss

from evenet.network.metrics.assigment import predict

from preprocessing.preprocess import unflatten_dict

import torch
import torch.nn as nn
from collections import defaultdict
from evenet.network.metrics.classification import ClassificationMetrics
from matplotlib import pyplot as plt


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),  # input dim 1 → hidden dim 8
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)  # hidden dim 8 → output dim 1
        )

    def forward(self, x):
        return self.net(x)

    def shared_step(self, x, batch_size):
        output = self.forward(x["conditions"])
        return {"classification": {"signal": output},
                "regression": {"signal": output}}


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
## Debug configuration ##
########################

wandb_enable = False
n_epoch = 10
debugger_enable = False

global_config.load_yaml("/Users/avencastmini/PycharmProjects/EveNet/share/local_test.yaml")
num_classes = global_config.event_info.class_label['EVENT']['signal'][0]

shape_metadata = json.load(
    open("/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output/shape_metadata.json"))

normalization_dict = torch.load("/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output/normalization.pt")

# Load the Parquet file locally
df = pq.read_table(
    "/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output/data_run_yulei_11.parquet").to_pandas()
# Optional: Subsample for speed

df.sample(frac=1).reset_index(drop=True)
df_number = len(df) // 2
df = df.head(df_number)

# Assignment setting
# Record attention head basic information
event_info = global_config.event_info
permutation_indices = dict()
num_targets = dict()
event_permutation = dict()
event_particles = dict()
import re
from evenet.utilities.group_theory import complete_indices, symmetry_group
for process in global_config.event_info.process_names:
    permutation_indices[process] = []
    num_targets[process] = []
    for event_particle_name, product_symmetry in global_config.event_info.product_symmetries[process].items():
        topology_name = ''.join(global_config.event_info.product_particles[process][event_particle_name].names)
        topology_name = f"{event_particle_name}/{topology_name}"
        topology_name = re.sub(r'\d+', '', topology_name)
        topology_category_name = global_config.event_info.pairing_topology[topology_name]["pairing_topology_category"]
        permutation_indices_tmp =  complete_indices(
            global_config.event_info.pairing_topology_category[topology_category_name]["product_symmetry"].degree,
            global_config.event_info.pairing_topology_category[topology_category_name]["product_symmetry"].permutations
            )
        permutation_indices[process].append(permutation_indices_tmp)
        event_particles[process] = [p for p in event_info.event_particles[process].names]
        event_permutation[process] = complete_indices(
            event_info.event_symmetries[process].degree,
            event_info.event_symmetries[process].permutations
        )
        permutation_group = symmetry_group(permutation_indices_tmp)
        num_targets[process].append(global_config.event_info.pairing_topology_category[topology_category_name]["product_symmetry"].degree)


# Convert to dict-of-arrays if needed
batch = {col: df[col].to_numpy() for col in df.columns}

# Preprocess batch
processed_batch = process_event_batch(batch, shape_metadata=shape_metadata, unflatten=unflatten_dict)

# Convert to torch
torch_batch = convert_batch_to_torch_tensor(processed_batch)

# with open("/Users/avencastmini/PycharmProjects/EveNet/workspace/normalization_file/PreTrain_norm.pkl", 'rb') as f:
#     normalization_file = pickle.load(f)

num_splits = df_number // 128

# Run forward
model = EveNetModel(
    config=global_config,
    device=torch.device("cpu"),
    normalization_dict=normalization_dict,
    assignment=True
)

model.freeze_module("Classification", global_config.options.Training.Components.Classification.get("freeze", {}))
model.freeze_module("Regression", global_config.options.Training.Components.Regression.get("freeze", {}))

# model = MLP()

model.train()

confusion_accumulator = ClassificationMetrics(
    num_classes=len(num_classes),
    normalize=True,
    device=torch.device("cpu"),
)

debugger = DebugHookManager(track_forward=True, track_backward=True, save_values=True)
if debugger_enable:
    debugger.attach_hooks(model)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

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
    # if (n_epoch > 2): model.eval()
    for i, batch in enumerate(split_batches):
        with torch.autograd.set_detect_anomaly(True):
        # if True:
            for name, tensor in batch.items():
                if torch.isnan(tensor).any():
                    print(f"[Epoch {iepoch} / Batch {i}] NaN found in input tensor: {name}")

            outputs = model.shared_step(batch, batch_size=len(batch["classification"]))
            # outputs = model.shared_step(batch)

            # Regression
            # reg_output = outputs["regression"]
            # flattened = torch.cat([v.squeeze(0) for v in reg_output.values()], dim=-1)
            # if torch.isnan(flattened).any():
            #     print(f"[Epoch {iepoch} / Batch {i}] NaN in regression output")
            #
            # reg_target = batch["regression-data"].float()
            # reg_mask = batch["regression-mask"].float()
            #
            # r_loss = reg_loss(predict=flattened, target=reg_target, mask=reg_mask)
            # if torch.isnan(r_loss).any():
            #     print(f"[Epoch {iepoch} /Batch {i}] NaN in regression loss")

            # Classification
            cls_output = next(iter(outputs["classification"].values()))
            cls_target = batch["classification"]

            print(cls_target[0], cls_output[0])
            # cls_output = torch.nn.Softmax(dim=1)(cls_output)
            preds = cls_output.argmax(dim=-1)

            print("target", torch.unique(cls_target, return_counts=True))
            print("pred", torch.unique(preds, return_counts=True))


            c_loss = cls_loss(predict=cls_output, target=cls_target, class_weight=normalization_dict['class_balance'])
            # c_loss = cls_loss(predict=cls_output, target=cls_target, class_weight=None)

            # mse_loss = torch.nn.MSELoss(reduction='none')
            # c_loss = mse_loss(cls_output, torch.nn.functional.one_hot(cls_target, num_classes=9).float())
            # print(f"c_loss", c_loss.item())
            if torch.isnan(c_loss).any():
                print(f"[Epoch {iepoch} / Batch {i}] NaN in classification loss")

            total_loss = c_loss.mean()  # + r_loss.mean() * 0.1

            symmetric_losses = ass_loss(
                    assignments = outputs["assignments"],
                    detections = outputs["detections"],
                    targets = batch["assignments-indices"],
                    targets_mask = batch["assignments-mask"],
                    num_targets = num_targets,
                    event_particles = event_particles,
                    event_permutations = event_permutation,
                    focal_gamma =  0.1,
            )

            assignment_predict = dict()
            for process in global_config.event_info.process_names:

                total_loss += symmetric_losses["assignment"][process]
                total_loss += symmetric_losses["detection"][process]

                assignment_predict[process] = predict(
                    assignments = outputs["assignments"][process],
                    detections=outputs["detections"][process],
                    product_symbolic_groups=event_info.product_symbolic_groups[process],
                    event_permutations=event_info.event_permutation[process],
                )


            # black_list = ["WJetsToQQ", "ZJetsToLL"]
            # process_name = global_config.event_info.process_names[i]
            # if process_name in black_list:
            #     process_name = global_config.event_info.process_names[i+1]
            # total_loss += symmetric_losses["assignment"][process_name]
            # print(f"{process_name}: {symmetric_losses["assignment"][process_name]}")

            print(f"[Epoch {iepoch} / Batch {i}] Total loss: {total_loss}", flush=True)

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"[Epoch {iepoch} / Batch {i}] Done")
            if wandb_enable:
                wandb.log(
                    {
                        "epoch": iepoch,
                        "total_loss": total_loss.item(),
                        "classification_loss": c_loss.mean().item(),
                        # "regression_loss": r_loss.mean().item()
                    }
                )
                for process in global_config.event_info.process_names:
                    wandb.log(
                        {
                            f"assignment_loss/{process}": symmetric_losses["assignment"][process].item(),
                            f"detection_loss/{process}": symmetric_losses["detection"][process].item()
                        }
                    )
            confusion_accumulator.update(
                cls_target,
                cls_output
            )

    if wandb_enable:
        fig = confusion_accumulator.plot_cm(class_names=num_classes)
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

    confusion_accumulator.reset()

    # break

if debugger_enable:
    debugger.remove_hooks()
    debugger.dump_debug_data()
