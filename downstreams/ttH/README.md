
To run data preprocessing, go to main directory and run the following command:

_It shares the same data preprocessing script with ttbar_semileptonic_

```bash
#!/bin/bash

PREPROCESS_SCRIPT="downstreams/ttbar_semileptonic/preprocess/preprocess_spanet.py"
YAML_CONFIG="downstreams/ttH/preprocess.yaml"
IN_BASE="/pscratch/sd/a/avencast/Event_Level_Analysis/Pretrain_Parquet/SPANet/ttH"
OUT_BASE="$PSCRATCH/Event_Level_Analysis/Pretrain_Parquet/SPANet/ttH"

#for dataset in train test train.sb test.sb; do
for dataset in train; do
    shifter python3 "$PREPROCESS_SCRIPT" "$YAML_CONFIG" \
        --in_dir "$IN_BASE/$dataset" \
        --store_dir "$OUT_BASE/$dataset"
done
```

To run the training, go to main directory and run the following command:
```bash
python evenet/train.py downstreams/ttbar_semileptonic/train.yaml --load_all
```

To run the evaluation, go to main directory and run the following command:
```bash
python evenet/predict.py downstreams/ttbar_semileptonic/predict.yaml
```