import argparse
import logging
import shutil
from pathlib import Path

import ray
import ray.data


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Globally shuffle a large Parquet dataset with Ray.")
    parser.add_argument("n_cpus", type=int, help="Number of CPUs to use")
    parser.add_argument("input_folder", type=Path, help="Path to input folder containing Parquet files")
    parser.add_argument("n_parts", type=int, help="Number of output parts to save the shuffled dataset")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    input_folder = args.input_folder.resolve()
    output_folder = input_folder.parent / f"{input_folder.name}_shuffled"
    output_folder.mkdir(parents=True, exist_ok=True)

    logging.info(f"Initializing Ray with {args.n_cpus} CPUs...")
    ray.init(
        num_cpus=args.n_cpus,
        _memory=250 * 1024 * 1024 * 1024,  # 30 GB object store
        runtime_env={"env_vars": {"RAY_memory_monitor_refresh_ms": "5000"}},
        # include_dashboard=True,
    )

    # List all Parquet files
    parquet_files = sorted([f for f in input_folder.glob("*.parquet")])
    if not parquet_files:
        raise RuntimeError(f"No .parquet files found in {input_folder}")
    logging.info(f"Found {len(parquet_files)} parquet files.")

    # Copy non-Parquet files to output
    non_parquet_files = [f for f in input_folder.iterdir() if f.is_file() and f.suffix != ".parquet"]
    for f in non_parquet_files:
        shutil.copy(f, output_folder)
        logging.info(f"Copied non-parquet file: {f.name}")

    # Read and shuffle dataset
    logging.info("Reading dataset into Ray...")
    ds = ray.data.read_parquet(
        [str(f) for f in parquet_files],
        # override_num_blocks=len(parquet_files) * platform_info.number_of_workers,
        ray_remote_args={
            "num_cpus": 0.5,
        },
    )
    logging.info(f"Dataset loaded with {ds.count()} rows")

    logging.info("Shuffling dataset globally...")
    ds_shuffled = ds.random_shuffle()

    logging.info(f"Repartitioning into {args.n_parts} parts...")
    ds_shuffled = ds_shuffled.repartition(args.n_parts)

    logging.info(f"Saving shuffled dataset to: {output_folder}")
    ds_shuffled.write_parquet(str(output_folder))

    stats = ray.data.DataContext.get_current().stats
    logging.info("Shuffling and saving complete.")
    logging.info(f"Dataset stats:\n{stats}")

    ray.shutdown()


if __name__ == "__main__":
    main()
