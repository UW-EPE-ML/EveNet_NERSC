# diHiggs to Multilepton Analysis


**Global Shuffle**
  ```bash
    python3 NERSC/count.py
  ```


To run data preprocessing, go to main directory and run the following command:

```bash
#!/bin/bash

PREPROCESS_SCRIPT="downstreams/HH_WWWW/preprocess/preprocess_HHML.py"
YAML_CONFIG="downstreams/HH_WWWW"
IN_BASE="/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/HHML_Combined_Balanced"
OUT_BASE="$PSCRATCH/Event_Level_Analysis/Pretrain_Parquet/HHML"

for dataset in train test; do
#for dataset in train; do
    python3 "$PREPROCESS_SCRIPT" "$YAML_CONFIG"/preprocess_${dataset}.yaml \
        --in_dir "$IN_BASE/$dataset" \
        --store_dir "$OUT_BASE/$dataset"
done
```


To run the training, go to main directory and run the following command:
```
python evenet/train.py downstreams/nu2flow/train.yaml --load_all
```

To run the evaluation, go to main directory and run the following command:
```
python evenet/predict.py downstreams/nu2flow/predict.yaml
```
