# Neutrino Generation Study

- reference link: https://zenodo.org/records/8113516
- dataset link: https://zenodo.org/api/records/8113516/files-archive

```bash
wget --content-disposition "https://zenodo.org/api/records/8113516/files-archive"
```

To run data preprocessing, go to main directory and run the following command:

```bash
#!/bin/bash

PREPROCESS_SCRIPT="downstreams/nu2flow/preprocess/preprocess_nu2flow.py"
YAML_CONFIG="downstreams/nu2flow/preprocess.yaml"
IN_BASE="/global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/nu2flow"
OUT_BASE="$PSCRATCH/Event_Level_Analysis/Pretrain_Parquet/nu2flow"

for dataset in mg5_test mg5_train pythia_test pythia_train; do
#for dataset in train; do
    shifter python3 "$PREPROCESS_SCRIPT" "$YAML_CONFIG" \
        --in_dir "$IN_BASE/$dataset" \
        --store_dir "$OUT_BASE/$dataset"
done
```


To run the training, go to main directory and run the following command:
```
python evenet/train.py downstreams/ttbar_semileptonic/train.yaml --load_all
```

To run the evaluation, go to main directory and run the following command:
```
python evenet/predict.py downstreams/ttbar_semileptonic/predict.yaml
```