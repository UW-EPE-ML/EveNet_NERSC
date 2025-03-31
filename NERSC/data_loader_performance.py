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
    ray.init(
        # num_cpus=10,
        # object_store_memory=10 * 1024 * 1024,
    )

    # Step 1: Read the parquet files
    base_dir = Path("/global/cfs/cdirs/m2616/tihsu/PreTrain_Parquet")

    parquet_files = [
        base_dir / file for file in base_dir.glob("*.parquet")
    ]

    ds = ray.data.read_parquet(parquet_files)

