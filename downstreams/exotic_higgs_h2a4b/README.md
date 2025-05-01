# Exotic Higgs 

signal processes is H->2a->4b, while mass of a spans from [20, 30, 40, 60] GeV.

To run data preprocessing, go to main directory and run the following command:

```bash
#!/bin/bash

PREPROCESS_SCRIPT="preprocessing/preprocess"
IN_BASE="/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/nu2flow"
OUT_BASE="$PSCRATCH/Event_Level_Analysis/Pretrain_Parquet/nu2flow"

for dataset in 20 30 40 50; do
#for dataset in train; do
    YAML_CONFIG="downstreams/exotic_higgs_h2a4b/preprocess_${mass}.yaml"
    shifter python3 "$PREPROCESS_SCRIPT" "$YAML_CONFIG" \
        --in_dir "$IN_BASE/$dataset" \
        --store_dir "$OUT_BASE/$dataset"
done
```