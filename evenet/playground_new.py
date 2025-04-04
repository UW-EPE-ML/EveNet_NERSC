import json

import pyarrow.parquet as pq
import pandas as pd

from evenet.control.config import config
from evenet.dataset.preprocess import process_event_batch, convert_batch_to_torch_tensor
from evenet.network.evenet_model import EvenetModel

from preprocessing.preprocess import unflatten_dict


config.load_yaml("/Users/avencastmini/PycharmProjects/EveNet/share/default.yaml")

shape_metadata = json.load(
    open("/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output/shape_metadata.json"))

# Load the Parquet file locally
df = pq.read_table("/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output/data_run_yulei_11.parquet").to_pandas()

# Optional: Subsample for speed
df = df.head(10)

# Convert to dict-of-arrays if needed
batch = {col: df[col].to_numpy() for col in df.columns}

# Preprocess batch
processed_batch = process_event_batch(batch, shape_metadata=shape_metadata, unflatten=unflatten_dict)

# Convert to torch
torch_batch = convert_batch_to_torch_tensor(processed_batch)

# Run forward
model = EvenetModel(config=config)
outputs = model.shared_step(torch_batch, batch_size=len(df))
