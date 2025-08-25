import numpy as np
import os
import torch
from torch import optim

from tqdm import tqdm
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot



from helpers.utils import EarlyStopping


def train_flow_asymm_SB(flow, hyperparameters_dict, device, SB1_train_dataset, SB1_val_dataset, SB2_train_dataset, SB2_val_dataset, flow_training_dir, early_stop=True, early_stop_patience=5, seed=2515):
    
    n_epochs = hyperparameters_dict["n_epochs"]
    lr = hyperparameters_dict["lr"]
    weight_decay = hyperparameters_dict["weight_decay"]
    batch_size = hyperparameters_dict["batch_size"]
    
    config_string = f"epochs{n_epochs}_lr{lr}_wd{weight_decay}_bs{batch_size}"
    checkpoint_path = os.path.join(flow_training_dir, f"{config_string}")

    # send network to device
    flow.to(device)

    optimizer = optim.AdamW(flow.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min = 0)
    cos_anneal_sched = True
    
    # set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # live loss plotting
    groups = {'loss': ['loss', 'val_loss']}
    plotlosses = PlotLosses(groups = groups)
    
    epochs = []
    losses, losses_val = [], []
    
    print("Training flow for", n_epochs, "epochs ...")
    print()
        
    if early_stop:
        early_stopping = EarlyStopping(patience = early_stop_patience)
        
    # save the best model
    val_loss_to_beat = 1e10
    best_epoch = -1
    
    # calculate the weights for the train and val datasets
    weight_SB1_train = 1. / SB1_train_dataset.shape[0]
    weight_SB1_val = 1. / SB1_val_dataset.shape[0]
    weight_SB2_train = 1. / SB2_train_dataset.shape[0]
    weight_SB2_val = 1. / SB2_val_dataset.shape[0]
        
    weights_train = [weight_SB1_train for i in range(SB1_train_dataset.shape[0])] + [weight_SB2_train for i in range(SB2_train_dataset.shape[0])]
    weights_val = [weight_SB1_val for i in range(SB1_val_dataset.shape[0])] + [weight_SB2_val for i in range(SB2_val_dataset.shape[0])]
    
    
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights_train, len(weights_train))
    sampler_val = torch.utils.data.sampler.WeightedRandomSampler(weights_val, len(weights_val))

    train_dataset = np.vstack((SB1_train_dataset, SB2_train_dataset))    
    val_dataset = np.vstack((SB1_val_dataset, SB2_val_dataset))

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers = 8, pin_memory = True, sampler = sampler_train)
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers = 8, sampler = sampler_val)
    
    for epoch in tqdm(range(n_epochs)):
        
        logs = {}
          
        losses_batch_per_e = []     
        for batch_ndx, data in enumerate(train_data):
            data = data.to(device)
            feats = data[:,:-1].float()
            cont = torch.reshape(data[:,-1], (-1, 1)).float()
            optimizer.zero_grad()     
            loss = -flow.log_prob(inputs=feats, context = cont).nanmean()  
            losses_batch_per_e.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()  
            
        if cos_anneal_sched:
            scheduler.step()
            
        # store the loss
        epochs.append(epoch)
        mean_loss = np.nanmean(losses_batch_per_e)
        losses.append(mean_loss)
        
        
        with torch.no_grad():

            val_losses_batch_per_e = []
            for batch_ndx, data in enumerate(val_data):
                data = data.to(device)
                feats = data[:,:-1].float()
                cont = torch.reshape(data[:,-1], (-1, 1)).float()
                val_loss = -flow.log_prob(inputs=feats, context = cont).mean()  
                val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())

            # store the loss
            mean_val_loss = np.nanmean(val_losses_batch_per_e)
            losses_val.append(mean_val_loss)
            
            # see if the model has the best val loss
            if mean_val_loss < val_loss_to_beat:
                val_loss_to_beat = mean_val_loss
                # save the model
                model_path = f"{checkpoint_path}_best_model.pt"
                torch.save(flow, model_path)
                best_epoch = epoch

            if early_stop:
                early_stopping(mean_val_loss)
             
        if early_stop:
            if early_stopping.early_stop:
                break
                
        plotlosses.update({
                'loss': mean_loss,
                'val_loss': mean_val_loss  
            })
        plotlosses.send()
                
    print("Done training!")
    
 
    return epochs, losses, losses_val, best_epoch
 
    
def train_flow(flow, hyperparameters_dict, device, train_dataset, val_dataset, flow_training_dir, flow_training_id, early_stop = True, seed = 2515, early_stop_patience=20):
    
    n_epochs = hyperparameters_dict["n_epochs"]
    lr = hyperparameters_dict["lr"]
    weight_decay = hyperparameters_dict["weight_decay"]
    batch_size = hyperparameters_dict["batch_size"]
    
    checkpoint_path = os.path.join(flow_training_dir, flow_training_id)

    # send network to device
    flow.to(device)

    optimizer = optim.AdamW(flow.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min = 0)
    cos_anneal_sched = True
    
    # set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    epochs = []
    losses, losses_val = [], []
    
    print("Training flow for", n_epochs, "epochs ...")
    print()
        
    if early_stop:
        early_stopping = EarlyStopping(patience = early_stop_patience)
        
    # save the best model
    val_loss_to_beat = 1e10
    best_epoch = -1
    
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 8, pin_memory = True)
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = True)
    
 
    for epoch in tqdm(range(n_epochs)):
        
          
        losses_batch_per_e = []     
        for batch_ndx, data in enumerate(train_data):
            data = data.to(device)
            feats = data[:,:-1].float()
            cont = torch.reshape(data[:,-1], (-1, 1)).float()
            optimizer.zero_grad()     
            loss = -flow.log_prob(inputs=feats, context = cont).nanmean()  
            losses_batch_per_e.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()  
            
        if cos_anneal_sched:
            scheduler.step()
            
        # store the loss
        epochs.append(epoch)
        mean_loss = np.nanmean(losses_batch_per_e)
        losses.append(mean_loss)
        
        
        
        with torch.no_grad():

            val_losses_batch_per_e = []
            for batch_ndx, data in enumerate(val_data):
                data = data.to(device)
                feats = data[:,:-1].float()
                cont = torch.reshape(data[:,-1], (-1, 1)).float()
                val_loss = -flow.log_prob(inputs=feats, context = cont).mean()  
                val_losses_batch_per_e.append(val_loss.detach().cpu().numpy())

            # store the loss
            mean_val_loss = np.nanmean(val_losses_batch_per_e)
            losses_val.append(mean_val_loss)

            # see if the model has the best val loss
            if mean_val_loss < val_loss_to_beat:
                val_loss_to_beat = mean_val_loss
                # save the model
                model_path = f"{checkpoint_path}_best_model.pt"
                torch.save(flow, model_path)
                best_epoch = epoch

            if early_stop:
                early_stopping(mean_val_loss)
             
        if early_stop:
            if early_stopping.early_stop:
                break
                
    print("Done training!")
    
 
    return epochs, losses, losses_val, best_epoch
 