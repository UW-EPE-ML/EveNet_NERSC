import json

import pyarrow.parquet as pq
import pandas as pd
import torch
from evenet.control.global_config import global_config
from evenet.dataset.preprocess import process_event_batch, convert_batch_to_torch_tensor
from evenet.network.evenet_model import EvenetModel

from preprocessing.preprocess import unflatten_dict

global_config.load_yaml("/Users/avencastmini/PycharmProjects/EveNet/share/local_test.yaml")

shape_metadata = json.load(
    open("/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output/shape_metadata.json"))

# Load the Parquet file locally
df = pq.read_table(
    "/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output/data_run_yulei_11.parquet").to_pandas()

# Optional: Subsample for speed
df = df.head(10)

# Convert to dict-of-arrays if needed
batch = {col: df[col].to_numpy() for col in df.columns}

# Preprocess batch
processed_batch = process_event_batch(batch, shape_metadata=shape_metadata, unflatten=unflatten_dict)

# Convert to torch
torch_batch = convert_batch_to_torch_tensor(processed_batch)

# Run forward
model = EvenetModel(config=global_config, device=torch.device("cpu"))
outputs = model.shared_step(torch_batch, batch_size=len(df))

from evenet.network.loss.classification import loss as cls_loss
from evenet.network.loss.regression import loss as reg_loss

cls_target = torch_batch["classification"]
reg_target = torch_batch["regression-data"].float()
reg_mask = torch_batch["regression-mask"].float()

reg_output = outputs["regression"]
flattened = torch.cat([v.squeeze(0) for v in reg_output.values()], dim=-1)
r_loss = reg_loss(predict=flattened, target=reg_target, mask=reg_mask)

cls_output = next(iter(outputs["classification"].values()))
c_loss = cls_loss(predict=cls_output, target=cls_target)
print(outputs)
