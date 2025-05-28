import argparse
import logging
import shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

import ray
import ray.data
from ray.data import Dataset


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Two-pass partial shuffle for large Parquet datasets.")
    parser.add_argument("n_cpus", type=int, help="Number of CPUs to use")
    parser.add_argument("input_folder", type=Path, help="Input folder containing Parquet files")
    parser.add_argument("--first_shuffle_percent", type=float, default=1.0,
                        help="First pass shuffle buffer size (percent of rows)")
    parser.add_argument("--second_shuffle_percent", type=float, default=5.0,
                        help="Second pass shuffle buffer size (percent of rows)")
    return parser.parse_args()


def copy_non_parquet_files(src: Path, dst: Path) -> None:
    for file in src.iterdir():
        if file.is_file() and file.suffix != ".parquet":
            shutil.copy(file, dst)
            logging.info(f"Copied non-parquet file: {file.name}")


def compute_buffer_sizes(ds: Dataset, first_pct: float, second_pct: float) -> tuple[int, int, int]:
    total_rows = ds.count()
    first_buf = int(total_rows * (first_pct / 100))
    second_buf = int(total_rows * (second_pct / 100))
    logging.info(f"Dataset has {total_rows:,} rows.")
    logging.info(f"Stage 1 buffer size: {first_buf:,} rows (~{first_pct}%)")
    logging.info(f"Stage 2 buffer size: {second_buf:,} rows (~{second_pct}%)")
    return total_rows, first_buf, second_buf


def save_batches(ds: Dataset, buffer_size: int, output_dir: Path, shuffle: bool=True) -> int:
    count = 0

    if shuffle:
        local_shuffle_buffer_size = buffer_size // 100
    else:
        local_shuffle_buffer_size = None

    for batch in ds.iter_batches(prefetch_batches=5, local_shuffle_buffer_size=local_shuffle_buffer_size, batch_size=buffer_size, batch_format="pandas"):
    # for batch in ds.iter_batches(prefetch_batches=5, local_shuffle_buffer_size=int(0.005 * buffer_size), batch_size=buffer_size):
    #     ray.data.from_pandas(batch).write_parquet(str(output_dir))
        output_path = output_dir / f"batch_{count:05d}.parquet"
        if not batch.empty:
            table = pa.Table.from_pandas(batch)
            pq.write_table(table, output_path)
            print(f"Saved batch {count} to {output_path}")
        # print(count)
        count += 1
    return count


def main():
    setup_logging()
    args = parse_args()

    input_dir = args.input_folder.resolve()
    temp_dir = input_dir.parent / f"{input_dir.name}_shuffle_temp"
    output_dir = input_dir.parent / f"{input_dir.name}_shuffled"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ray.init(num_cpus=args.n_cpus, include_dashboard=True)
    logging.info(f"Shuffling data from {input_dir} to {output_dir} using {args.n_cpus} CPUs...")

    copy_non_parquet_files(input_dir, output_dir)

    parquet_files = list(input_dir.glob("*.parquet"))
    ds = ray.data.read_parquet(
        [str(f) for f in parquet_files], shuffle="files",
        ray_remote_args={"num_cpus": 1.0},
        # override_num_blocks=10,
    )
    total_rows, first_buffer, second_buffer = compute_buffer_sizes(
        ds, args.first_shuffle_percent, args.second_shuffle_percent
    )

    # logging.info("Stage 1: Partial shuffle and write to temp...")
    # stage1_parts = save_batches(ds, first_buffer, temp_dir)
    # logging.info(f"Stage 1 complete: wrote {stage1_parts} temp batches.")
    del ds  # Free memory after stage 1

    logging.info("Stage 2: Re-shuffle from temp and write final output...")
    temp_files = list(temp_dir.rglob("*.parquet"))
    ds2 = ray.data.read_parquet([str(f) for f in temp_files], shuffle="files", ray_remote_args={"num_cpus": 2.0})
    stage2_parts = save_batches(ds2, second_buffer, output_dir, shuffle=False)
    logging.info(f"Stage 2 complete: wrote {stage2_parts} final batches.")

    logging.info(f"✅ Final shuffled dataset saved to {output_dir}")
    ray.shutdown()


if __name__ == "__main__":
    main()
