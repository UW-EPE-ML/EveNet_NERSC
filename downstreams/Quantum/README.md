# Quantum Entanglement Study

## TT2L

To run data preprocessing, go to main directory and run the following command:

```bash
#!/bin/bash

PREPROCESS_SCRIPT="downstreams/Quantum/preprocess/preprocess_TT2L.py"
YAML_DIR="downstreams/Quantum/"
IN_BASE="/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/Quantum/TT2L"
OUT_BASE="$PSCRATCH/Event_Level_Analysis/Pretrain_Parquet/Quantum-TT2L"

for dataset in test; do
    shifter python3 "$PREPROCESS_SCRIPT" "$YAML_DIR/preprocess_TT2L_$dataset.yaml" \
        --pretrain_dirs "$IN_BASE/$dataset" \
        --store_dir "$OUT_BASE/$dataset" \
        --cpu_max 100
done
```


To run the training, go to main directory and run the following command:
```
python evenet/train.py downstreams/Quantum/train_TT2L.yaml --load_all
```

To run the evaluation, go to main directory and run the following command:
```
python evenet/predict.py downstreams/Quantum/predict_TT2L.yaml
```