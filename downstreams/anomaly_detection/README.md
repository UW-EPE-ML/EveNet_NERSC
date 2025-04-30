## Anomaly Detection

### Step1: Dataset preparation
#### From EveNet dataformat
To have a dataset in the EveNet data format for a specific task, 
you can first run the `preprocess.py` script. 
This script will generate a dataset in the EveNet data format,
it is okay to use the dummy dataset config as later stage, we will create the customized samples.
##### TODO: postprocessing raise error (no actual effect in the task)
```bash
# Background samples
python3 preprocessing/preprocess.py downstreams/anomaly_detection/preprocessing/preprocess_evenet.yaml --pretrain_dirs /global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/Run_2.Dec20/ /global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/Run_2.Dec21 /global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/Run_2.Dec22  --store_dir /pscratch/sd/t/tihsu/database/EveNet-AnomalyDetection/QCD
# Anomaly samples
python3 preprocessing/preprocess.py downstreams/anomaly_detection/preprocessing/preprocess_evenet.yaml --in_dir /pscratch/sd/t/tihsu/database/AnomalyDetection/Delphes_Sample/Zjets_m300/top/  --store_dir /pscratch/sd/t/tihsu/database/EveNet-AnomalyDetection/Zjet
```

To further produced side band and signal region data. Run:
```bash
python3 00prepare_toy_model_input.py config/workflow_toy_model.yaml
python3 00prepare_toy_model_input.py config/workflow_toy_model.yaml --no_signal
```
#### CMS real data
TODO
### Step2: Training
```bash
cd ../..
shifter python evenet/train.py downstreams/anomaly_detection/config/full_train.yaml  --load_all
```
