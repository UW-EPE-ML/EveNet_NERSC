import os
import argparse
import logging
import sys

import ray
from ray import air, tune
from ray.air import session
from ray.train.torch import TorchCheckpoint
from ray.tune.schedulers import AsyncHyperBandScheduler
import wandb
from functools import partial

from evenet.control.config import config
from evenet.dataset.preprocess import process_event_batch, process_event_batch_old
from evenet.network_scratch.evenet_model import EvenetModel

from preprocessing.preprocess import unflatten_dict
import json

from evenet.engine import EveNetEngine

def setup_logger(log_file: str = "output.log", level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)

    for handler in [file_handler, stream_handler]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def main(args):
    # os.environ["RAY_ENABLE_MAC_LARGE_OBJECT_STORE"] = "1"

    config.load_yaml(args.config)

    shape_metadata = json.load(open("/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output/shape_metadata.json"))

    ray.init(
        # num_cpus=10,
        # object_store_memory=10 * 1024 * 1024,
        # local_mode=True,
    )

    # for early stopping
    sched = AsyncHyperBandScheduler()

    ds = ray.data.read_parquet([
        "/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/test_output/data_run_yulei_11.parquet",
        # "/Users/avencastmini/PycharmProjects/EveNet/workspace/test_data/PreTrain_Parquet/multi_process_1.parquet"
    ])# .limit(200)

    # ds = ds.take_batch(10)

    # ds_test = process_event_batch(ds)

    model = EvenetModel(config = config)

    process_event_batch_partial = partial(process_event_batch, shape_metadata=shape_metadata, unflatten=unflatten_dict)

    processed_ds = ds.map_batches(
        process_event_batch_partial,
        # process_event_batch_old,
        # concurrency=5,
        # batch_format="default"
    )

    # Step 3: Apply transformation

    # model = JetReconstructionModel(config=config, torch_script=False, total_events=10000)
    for i, batch in enumerate(processed_ds.iter_torch_batches(batch_size=1024)):
        # Each batch is a list of tuples as returned above
        print("Batch ", i)

        model.training_step(batch)

        exit(1)

    ray.shutdown()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EveNet Training Program")
    parser.add_argument("config", help="Path to config file")

    args, _ = parser.parse_known_args()

    main(args)
