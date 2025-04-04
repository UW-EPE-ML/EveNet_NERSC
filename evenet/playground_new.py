import json
import math

import pyarrow.parquet as pq
import pandas as pd
import torch
from evenet.control.global_config import global_config
from evenet.dataset.preprocess import process_event_batch, convert_batch_to_torch_tensor
from evenet.network.evenet_model import EvenetModel

from evenet.network.loss.classification import loss as cls_loss
from evenet.network.loss.regression import loss as reg_loss

from preprocessing.preprocess import unflatten_dict

global_config.load_yaml("/Users/avencastmini/PycharmProjects/EveNet/share/local_test.yaml")

shape_metadata = json.load(
    open("/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output/shape_metadata.json"))

# Load the Parquet file locally
df = pq.read_table(
    "/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output/data_Run_2.Dec20_run_yulei_11.parquet").to_pandas()

# Optional: Subsample for speed
df = df.head(500)

# Convert to dict-of-arrays if needed
batch = {col: df[col].to_numpy() for col in df.columns}

# Preprocess batch
processed_batch = process_event_batch(batch, shape_metadata=shape_metadata, unflatten=unflatten_dict)

# Convert to torch
torch_batch = convert_batch_to_torch_tensor(processed_batch)

# Run forward
model = EvenetModel(config=global_config, device=torch.device("cpu"))
model.train()

def check_nan_hook(module, input, output):
    if isinstance(output, torch.Tensor) and torch.isnan(output).any():
        print(f"[NaN Detected] Forward output of {module.__class__.__name__}")
    elif isinstance(input, tuple):
        for i, inp in enumerate(input):
            if isinstance(inp, torch.Tensor) and torch.isnan(inp).any():
                print(f"[NaN Detected] Forward input {i} of {module.__class__.__name__}")

def check_grad_hook(param, name):
    def hook(grad):
        if grad is not None and torch.isnan(grad).any():
            print(f"[NaN Detected] Grad of {name}")
    return hook

def gelu_backward_hook(module, grad_input, grad_output):
    if any(torch.isnan(g).any() for g in grad_input):
        print(f"[NaN Detected] in backward of {module.__class__.__name__}")
    if any(torch.isinf(g).any() for g in grad_input):
        print(f"[Inf Detected] in backward of {module.__class__.__name__}")
    return grad_input

for name, module in model.named_modules():
    if isinstance(module, torch.nn.GELU):
        # print(f"Registering backward hook on {name}")
        module.register_full_backward_hook(gelu_backward_hook)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


batch_size = len(df)
num_splits = 10
split_batches = []

for i in range(num_splits):
    start = math.floor(i * batch_size / num_splits)
    end = math.floor((i + 1) * batch_size / num_splits)
    split = {k: v[start:end] for k, v in torch_batch.items()}
    split_batches.append(split)

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

        total_loss = c_loss.mean() + r_loss.mean() * 0

        print(f"[Batch {i}] Total loss: {total_loss}", flush=True)

        # Backprop
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"[Batch {i}] Done")

        # break