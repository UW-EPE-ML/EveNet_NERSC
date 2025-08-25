import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import torch
import os

import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from tqdm import tqdm
from helpers.utils import EarlyStopping

from sklearn.preprocessing import StandardScaler


def np_to_torch(array, device):
    
    return torch.tensor(array.astype(np.float32)).to(device)
    
"""
NEURAL NET
"""

class NeuralNet(nn.Module):
    def __init__(self, layers, n_inputs):
        super(NeuralNet, self).__init__()

        self.layers = []
        for nodes in layers:
            self.layers.append(nn.Linear(n_inputs, nodes))
            self.layers.append(nn.ReLU())
            n_inputs = nodes
        self.layers.append(nn.Linear(n_inputs, 1))
        self.layers.append(nn.Sigmoid())
        self.model_stack = nn.Sequential(*self.layers)

        
    def forward(self, x):
        
        return self.model_stack(x)
    
    
    
def train_NN(X_train, Y_train, w_train, X_val, Y_val, w_val, seed, layers, hyperparameters_dict, device, results_path = "./loc_nn_model"):
    
    n_epochs = hyperparameters_dict["n_epochs"]
    batch_size = hyperparameters_dict["batch_size"]
    lr = hyperparameters_dict["lr"]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    loc_scaler = StandardScaler().fit(X_train)
    X_train = loc_scaler.transform(X_train)
    X_val = loc_scaler.transform(X_val)

    # send to device
    X_train = np_to_torch(X_train, device)
    Y_train = np_to_torch(Y_train, device)
    w_train = np_to_torch(w_train, device)
    
    X_val = np_to_torch(X_val, device)
    Y_val = np_to_torch(Y_val, device)
    w_val = np_to_torch(w_val, device)
    

    train_set = torch.utils.data.TensorDataset(X_train, Y_train, w_train)
    val_set = torch.utils.data.TensorDataset(X_val, Y_val, w_val)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False)

    # initialze the network
    dense_net = NeuralNet(layers = layers, n_inputs = X_train.shape[1])
    criterion = F.binary_cross_entropy 
    optimizer = torch.optim.Adam(dense_net.parameters(), lr=lr)
    dense_net.to(device)

    early_stopping = EarlyStopping()

     # save the best model
    val_loss_to_beat = 1e10
    best_epoch = -1

    epochs, losses, losses_val = [], [], []

    for epoch in tqdm(range(n_epochs)):
        losses_batch_per_e = []
        # batching    
        for batch_index, (batch_data, batch_labels, batch_weights) in enumerate(train_loader):

            # calculate the loss, backpropagate
            optimizer.zero_grad()
            loss = criterion(dense_net(batch_data), batch_labels, weight = batch_weights)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses_batch_per_e.append(loss.detach().cpu().numpy())

        epochs.append(epoch)
        losses.append(np.mean(losses_batch_per_e))

        # validation
        with torch.no_grad():
            val_losses_batch_per_e = []

            for batch_index, (batch_data, batch_labels, batch_weights) in enumerate(val_loader):
                # calculate the loss, backpropagate
                optimizer.zero_grad()

                val_loss = criterion(dense_net(batch_data), batch_labels, weight = batch_weights) 
                val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())

            losses_val.append(np.mean(val_losses_batch_per_e))

            # see if the model has the best val loss
            if np.mean(val_losses_batch_per_e) < val_loss_to_beat:
                val_loss_to_beat = np.mean(val_losses_batch_per_e)
                # save the model
                model_path = f"{results_path}.pt"
                torch.save(dense_net, model_path)
                best_epoch = epoch

            early_stopping(np.mean(val_losses_batch_per_e))

        if early_stopping.early_stop:
            break



    return loc_scaler, losses, losses_val, best_epoch

