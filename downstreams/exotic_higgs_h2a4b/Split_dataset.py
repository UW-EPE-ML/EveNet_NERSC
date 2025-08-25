import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
import shutil
import random
import h5py
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from functools import partial
from tqdm import tqdm

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Split h5 files into train/test sets.")
    parser.add_argument("root_dir", type=str, help="Root directory to search (contains dir/run*/{process}.h5)", nargs='+')
    parser.add_argument("--train_ratio", type=float, default=0.5, help="Proportion of entries for training (default: 0.8)")
    parser.add_argument("--limit", type=int, default=None, help="Max entries per process to use (default: all)")
    parser.add_argument("--output_dir", type=str, default="split_output", help="Output directory to store train/test dirs")
    parser.add_argument("-c", "--cpus", type=int, default=os.cpu_count(), help="Number of concurrent threads for file operations (default: 8)")
    return parser.parse_args()

def find_h5_files(root_dirs):
    files = []
    for root in root_dirs:
        files.extend(Path(root).rglob("*/*.h5"))

    return files
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import random

def get_entry_count_safe(path):
    try:
        with h5py.File(path, 'r') as f:
            first_key = next(iter(f.keys()))
            return (path, len(f[first_key]))
    except Exception as e:
        console.print(f"[red]Failed to read {path}: {e}[/red]")
        return (path, 0)

def split_files(files, train_ratio, limit, max_workers=8):
    # Group by process name
    process_groups = defaultdict(list)
    for f in files:
        process = '_'.join(f.stem.split("_")[:-1])
        process_groups[process].append(f)

    # Precompute entry counts using multithreading and show progress
    all_paths = [f for group in process_groups.values() for f in group]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(get_entry_count_safe, all_paths),
            total=len(all_paths),
            desc="Counting entries"
        ))

    # Build lookup: path -> count
    entry_counts = {path: count for path, count in results}

    split_result = {}

    for process in tqdm(process_groups, desc="Processing groups"):
        paths = process_groups[process]
        random.shuffle(paths)
        total_entries = 0
        train_entries = 0
        test_entries = 0
        selected_files = []

        for p in paths:
            count = entry_counts.get(p, 0)
            if limit is not None and total_entries + count > limit:
                remaining = limit - total_entries
                if remaining > 0:
                    selected_files.append((p, remaining))  # mark partial
                    total_entries += remaining
                break
            else:
                selected_files.append((p, count))
                total_entries += count

        train_cut = int(total_entries * train_ratio)
        acc = 0
        train_files = []
        test_files = []

        for path, count in selected_files:
            if acc < train_cut:
                train_files.append(path)
                train_entries += count
            else:
                test_files.append(path)
                test_entries += count
            acc += count

        split_result[process] = {
            "train": train_files,
            "test": test_files,
            "total": total_entries,
            "nTrain": train_entries,
            "nTest": test_entries,
        }

    return split_result



def copy_single_file(src, dest_base, split):
    rel_path = src.relative_to(src.parents[2])  # run*/{process}.parquet
    dest = Path(dest_base) / split / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return str(dest)

def copy_files(split_result, out_dir, root_dirs, max_workers=8):
    tasks = []

    for process, splits in split_result.items():
        for split in ["train", "test"]:
            for src in splits[split]:
                tasks.append((src, split))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(copy_single_file, src, out_dir, split): (src, split)
            for src, split in tasks
        }

        for future in track(as_completed(futures), total=len(futures), description="Copying files..."):
            try:
                results.append(future.result())
            except Exception as e:
                src, split = futures[future]
                print(f"Failed to copy {src} for split {split}: {e}")

    return results
def display_summary(results, json_path=None):
    console = Console()
    table = Table(title="HDF5 Process Split Summary")

    table.add_column("Process", style="cyan")
    table.add_column("Total Entries", justify="right")
    table.add_column("Train Entries", justify="right")
    table.add_column("Test Entries", justify="right")
    table.add_column("Train Files", justify="right")
    table.add_column("Test Files", justify="right")

    summary = {}

    for process, data in results.items():
        # Count entries in train/test

        table.add_row(
            process,
            str(data["total"]),
            str(data["nTrain"]),
            str(data["nTest"]),
            str(len(data["train"])),
            str(len(data["test"]))
        )

        summary[process] = {
            "total_entries": data["total"],
            "train_entries": data["nTrain"],
            "test_entries": data["nTest"],
            "train_files": [str(f) for f in data["train"]],
            "test_files": [str(f) for f in data["test"]],
        }

    console.print(table)

    if json_path:
        import json
        write_out_summary = dict()
        for process, data in summary.items():
            write_out_summary[process] = {
                "total_entries": data["total_entries"],
                "train_entries": data["train_entries"],
                "test_entries": data["test_entries"],
            }

        with open(json_path, "w") as f:
            json.dump(write_out_summary, f, indent=2)
        console.print(f"[green]Summary saved to {json_path}[/green]")

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = find_h5_files(args.root_dir)
    if not files:
        console.print("[bold red]No h5 files found![/bold red]")
        return

    console.print(f"[green]Found {len(files)} h5 files. Processing...[/green]")

    split_result = split_files(files, args.train_ratio, args.limit, max_workers=args.cpus)
    display_summary(split_result, json_path=os.path.join(args.output_dir, "split_summary.json"))

    console.print(f"[blue]Copying files to output directory: {args.output_dir}[/blue]")
    copy_files(split_result, args.output_dir, args.root_dir, max_workers=args.cpus)
    console.print("[bold green]Done![/bold green]")

if __name__ == "__main__":
    main()
