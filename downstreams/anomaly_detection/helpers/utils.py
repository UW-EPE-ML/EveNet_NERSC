"""
NOT MY CODE -- minimally adapted from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
"""

import torch, os, json
from preprocessing.preprocess import unflatten_dict, flatten_dict
import pyarrow.parquet as pq

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
     
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, verbose = False):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
    def __call__(self, val_loss):
        
        if self.best_loss == None:
            self.best_loss = val_loss
        
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            #print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print('INFO: Early stopping')
                self.early_stop = True

def save_file(
    save_dir,
    data_df,
    norm_dict,
    event_filter,
    postfix=""
):

    os.makedirs(save_dir, exist_ok=True)
    if event_filter is not None:
        filtered_df = {col: data_df[col][event_filter.to_numpy()] for col in data_df}
    else:
        filtered_df = data_df
    flatten_data, meta_data = flatten_dict(filtered_df)

    ### Save to parquet
    pq.write_table(flatten_data, f"{save_dir}/data{postfix}.parquet")

    with open(f"{save_dir}/shape_metadata{postfix}.json", "w") as f:
        json.dump(meta_data, f)

    print(f"[INFO] Final table size: {flatten_data.nbytes / 1024 / 1024:.2f} MB")
    print(f"[Saving] Saving {flatten_data.num_rows} rows to {save_dir}/data.parquet")

    torch.save(norm_dict, f"{save_dir}/normalization{postfix}.pt")