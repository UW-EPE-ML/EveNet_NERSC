import os
import argparse
import logging
import sys
from pathlib import Path

import ray
from ray import air, tune
from ray.air import session
from ray.train.torch import TorchCheckpoint
from ray.tune.schedulers import AsyncHyperBandScheduler

from evenet.dataset.preprocess import process_event_batch

if __name__ == '__main__':
    # Step 1: Read the parquet files
    base_dir = Path("/global/cfs/cdirs/m2616/tihsu/PreTrain_Parquet")

    parquet_files = [
        str(base_dir / file) for file in base_dir.glob("*.parquet")
    ]

    ray.init(
        num_cpus=len(parquet_files),
        # object_store_memory=10 * 1024 * 1024,
    )

    ds = ray.data.read_parquet(parquet_files)

    # Step 2: Process the data
    processed_ds = ds.map(
        process_event_batch,
        # concurrency=5,
        # batch_format="default"
    )

    print(processed_ds.stats)

    for i, batch in enumerate(processed_ds.iter_batches(batch_size=1024)):
        # Each batch is a list of tuples as returned above
        print("Batch ", i)
