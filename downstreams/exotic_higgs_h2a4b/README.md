# Exotic Higgs 

signal processes is H->2a->4b, while mass of a spans from [20, 30, 40, 60] GeV.

To run data preprocessing, go to main directory and run the following command:

```bash
#!/bin/bash

PREPROCESS_SCRIPT="preprocessing/preprocess.py"
IN_BASE="/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data"
OUT_BASE="$PSCRATCH/Event_Level_Analysis/Pretrain_Parquet/h2a4b"

#for mass in 20 30 40 50; do
for mass in 20; do
  for dataset in train test; do
  
    YAML_CONFIG="downstreams/exotic_higgs_h2a4b/preprocess_${mass}.yaml"
    shifter python3 "$PREPROCESS_SCRIPT" "$YAML_CONFIG" \
        --pretrain_dirs "$IN_BASE/Run_2.Haa.20250114.${dataset}" "$IN_BASE/Run_2.QCD_extra.20250205.${dataset}" \
        --store_dir "$OUT_BASE/${dataset}_$mass"
  done
done
```