import os
import h5py
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------

data_prefix = "/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data"

tags = [
    "Run_2.Dec20",
    "Run_2.Dec21",
    "Run_2.Dec22",
    "Run_2.Pretrain.20250505",
    "Run_2.Pretrain.20250507",
    "Run_3.Pretrain.20250527.Run2Extra"
]

INPUT_ROOTS = [Path(data_prefix) / tag for tag in tags]
OUTPUT_ROOT = Path(data_prefix) / "Combined_Balanced"

NUM_RUNS = 500
SEED = 42

random.seed(SEED)

# ---------------------------
# 1️⃣ Scan all .h5 files from all input roots
# ---------------------------
print(f"Scanning all .h5 files in {len(INPUT_ROOTS)} roots...")

process_files = defaultdict(list)  # process_name -> list of (Path, entries)

for root in INPUT_ROOTS:
    print(f"  Scanning {root} ...")
    for run_dir in root.iterdir():
        if run_dir.is_dir():
            for file in run_dir.glob("*.h5"):
                try:
                    with h5py.File(file, 'r') as f:
                        first_key = list(f.keys())[0]
                        n_entries = f[first_key].shape[0]
                    process_name = "_".join(file.stem.split("_")[:-1])
                    process_files[process_name].append( (file, n_entries) )
                except Exception as e:
                    print(f"⚠️ Could not read {file}: {e}")

print(f"✔️ Found {len(process_files)} unique processes across all roots")

# ---------------------------
# 2️⃣ Check: assert enough files per process
# ---------------------------
print("Checking file counts per process...")
for process, files in process_files.items():
    if len(files) < NUM_RUNS:
        print(f"⚠️ WARNING: Process '{process}' has only {len(files)} files, but {NUM_RUNS} runs requested!")
        assert False, f"Process '{process}' has too few files: {len(files)} < {NUM_RUNS}"

# ---------------------------
# 3️⃣ Bin pack by total entries per process
# ---------------------------
print(f"Distributing files into {NUM_RUNS} runs by total entries...")

run_plan = defaultdict(list)   # run_idx -> list of (Path, entries)
entries_summary = defaultdict(lambda: defaultdict(int))  # run_idx -> process -> entries

for process, files in tqdm(process_files.items(), desc="Balancing processes"):
    files.sort(key=lambda x: -x[1])  # largest first
    bins = [ (0, []) for _ in range(NUM_RUNS) ]
    for file_path, n_entries in files:
        min_idx = min(range(NUM_RUNS), key=lambda i: bins[i][0])
        bins[min_idx] = (bins[min_idx][0] + n_entries, bins[min_idx][1] + [(file_path, n_entries)])
    for run_idx, (_, file_list) in enumerate(bins):
        run_plan[run_idx].extend(file_list)
        for _, n_entries in file_list:
            entries_summary[run_idx][process] += n_entries

# ---------------------------
# 4️⃣ Create output runs and symlinks
# ---------------------------
print(f"Creating balanced structure at: {OUTPUT_ROOT}")
OUTPUT_ROOT.mkdir(exist_ok=True)

for run_idx in tqdm(range(NUM_RUNS), desc="Creating runs"):
    run_dir = OUTPUT_ROOT / f"run_{run_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for file_path, _ in run_plan[run_idx]:
        link_name = run_dir / file_path.name
        try:
            os.symlink(file_path.resolve(), link_name)
        except FileExistsError:
            pass  # skip if rerun

# ---------------------------
# 5️⃣ Save summary CSV
# ---------------------------
print("Saving per-run, per-process entries summary...")

summary_df = pd.DataFrame.from_dict(entries_summary, orient="index").fillna(0).astype(int)
summary_df.index.name = "Run"
summary_df.to_csv(OUTPUT_ROOT / "entries_summary.csv")

print(f"✅ Done! Combined balanced runs created.")
print(f"📄 Summary CSV saved to: {OUTPUT_ROOT / 'entries_summary.csv'}")