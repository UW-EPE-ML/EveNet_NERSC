# 🧪 Data Preprocessing for EveNet

This preprocessing pipeline prepares structured event-level datasets for EveNet. It merges input .h5 files, flattens structured arrays into 2D pyarrow tables, calculates normalization statistics, and saves outputs as .parquet and .json files ready for model training.

## 📍 Folder Structure

The input directory should follow this structure:

```
/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/
├── Run_2.Dec20/
│   ├── run_1/
│   │   ├── process_1.h5
│   │   ├── process_2.h5
│   ├── run_2/
│   │   ├── ...
├── Run_2.Dec21/
│   ├── run_1/
│   │   ├── ...
...
```

Each Run folder (e.g., Run_2.Dec20) contains sub-run folders (run_1, run_2, …), which hold the .h5 files representing different processes.


## 🛠 What the Script Does

For every sub-run folder:    
1.	Merges all .h5 process files into a single structured data dictionary.  
2.	Converts structured data into a 2D pyarrow Table.  
3.	Saves the output as:  
    - data_<unique_id>.parquet (flattened table)  
    - shape_metadata.json (used for reconstruction)  
4.	Computes normalization statistics (mean, std) across all data.  
5.	Merges statistics from all sub-runs and saves a single .pt normalization file.  


## 🚀 Run on NERSC

- **Paths**
	- **Input root:** `/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data`
	- **Pretrain Run folders:** `Run_2.Dec20`, `Run_2.Dec21`, `Run_2.Dec22`
	- **Output directory:** better to say on `$SCRATCH` folder

- **Example Command**
    ```bash
    pretrain_dir="/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data"
    output_dir="$PSCRATCH/Event_Level_Analysis/Pretrain_Parquet/"

    python3 preprocessing/preprocess.py share/preprocess_pretrain.yaml \
        --c 25 \
        --pretrain_dirs \
            ${pretrain_dir}/Run_2.Dec20 \
            ${pretrain_dir}/Run_2.Dec21 \
            ${pretrain_dir}/Run_2.Dec22 \
            ${pretrain_dir}/Run_2.Pretrain.20250505 \
            ${pretrain_dir}/Run_2.Pretrain.20250507 \
            ${pretrain_dir}/Run_3.Pretrain.20250527.Run2Extra \
        --store_dir ${output_dir}/run.20250527.654M | tee ${output_dir}/run.20250527.654M.log
  
    shifter python3 preprocessing/preprocess.py share/preprocess_pretrain.yaml \
        --c 60 \
        --pretrain_dirs \
        ${pretrain_dir}/Combined_Balanced \
        --store_dir ${output_dir}/run.20250625.2700M
    ```

    This command will:
  - Process all sub-runs in the three Run folders
  - Generate flattened .parquet files and normalization .pt file
  - Save logs to run.20250403.log

- **Global Shuffle**
  ```bash
    python3 NERSC/global_shuffle.py --first_shuffle_percent 1.0 --second_shuffle_percent 0.5 25 \
    /pscratch/sd/a/avencast/Event_Level_Analysis/Pretrain_Parquet/run.20250527.654M/
  ```


## 📤 Output Files

Under the output directory (`--store_dir`), the script saves:

```
/run.20250403/
├── data_<timestamp>.parquet       # Flattened input table
├── shape_metadata.json            # Shape metadata for reconstruction
├── regression_norm.pt             # Normalization statistics
└── run.20250403.log               # Optional log if tee is used
```



## ⚠️ Notes
- You must supply a valid preprocess_config.yaml (e.g., `EveNet/share/preprocess_pretrain.yaml`).
- The script will skip any sub-run folder where no valid .h5 files are found or matching fails.
- All normalization is computed globally across all sub-runs and merged at the end.



