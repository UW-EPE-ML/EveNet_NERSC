
To run data preprocessing, go to main directory and run the following command:
```
# train
python3 downstreams/ttbar_semileptonic/preprocess/preprocess_spanet.py downstreams/ttbar_semileptonic/preprocess.yaml \
--in_dir /global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/SPANet/semi_leptonic_ttbar/train \
--store_dir $PSCRATCH/Event_Level_Analysis/Pretrain_Parquet/SPANet/ttbar_semi_leptonic/train

# test
python3 downstreams/ttbar_semileptonic/preprocess/preprocess_spanet.py downstreams/ttbar_semileptonic/preprocess.yaml \
--in_dir /global/cfs/cdirs/m2616/avencast/Event_Level_Analysis/data/SPANet/semi_leptonic_ttbar/test \
--store_dir $PSCRATCH/Event_Level_Analysis/Pretrain_Parquet/SPANet/ttbar_semi_leptonic/test
```

To run the training, go to main directory and run the following command:
```
python evenet/train.py downstreams/ttbar_semileptonic/train.yaml --load_all
```