from torch import nn, tensor, float32
import os
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import r2_score
import numpy as np
from models.viscosity_models import CNN3D
from pytorchtools import EarlyStopping
from typing import List, Optional, Callable, Tuple
from utils.datastruct import history, metrics
from tqdm import tqdm

def train(model : CNN3D, 
          data_loader : DataLoader, 
          optimizer : torch.optim.Optimizer,
          criterion : torch.nn.modules.loss._Loss, 
          device : torch.device) -> float:

    train_loss = []
    model.train()
    
    for (X,y) in data_loader:
        X = X.to(device)
        y = y.to(device)
        
        y = y.to(float32) 
        # zeroing grads
        optimizer.zero_grad()
        # model out
        #out = model(data.x, data.edge_index, data.batch)
        out = model(X)
        #loss = criterion(out,data.y.reshape(-1,1))
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    return np.mean(train_loss)


def test(model  : CNN3D, data_loader : DataLoader, criterion : torch.nn.modules.loss._Loss,  device : torch.device) -> Tuple[float, float]:
    model.eval()
    
    y_h_all = []
    y_all =[]
    test_loss = []
    for (X,y) in data_loader:
        X = X.to(device)
        
        y = y.to(float32) 
        
        y_h = model(X)  
        
        loss = criterion(y_h.detach().cpu(),y)
        test_loss.append(loss)
        
        y_h_all.extend(y_h.detach().cpu().numpy())
        y_all.extend(y.numpy())

    return (np.mean(test_loss), r2_score(np.array(y_all),np.array(y_h_all)))
    
    
    
def train_epochs(model : CNN3D , 
                 dataloaders : List[DataLoader], 
                 optimizer : torch.optim.Optimizer, ##Callable[torch.optim.Optimizer], 
                 criterion : torch.nn.modules.loss._Loss, #Callable[], 
                 epochs : int, 
                 early_stop : Optional[int],
                 device : torch.device,
                 path : str,
                 save_weights_frequency : int) -> Tuple[CNN3D, history]:
  
    # parse dataloaders
    '''
    if len(data_loader)>2 :
        (train_loader, val_loader, test_loader) = dataloaders
    else : 
        (train_loader, val_loader) = dataloaders
    '''
    (train_loader, val_loader, test_loader) = dataloaders
    
    if early_stop : early_stopping = EarlyStopping(patience=early_stop, verbose=True)
    
    train_loss_list=[]
    val_loss_list=[]
    test_loss_list=[]
    
    train_r2_list=[]
    val_r2_list=[]
    test_r2_list=[]
    
    for epoch in tqdm(range(epochs)):

        loss = train(model,
                     data_loader=train_loader,
                     optimizer=optimizer,
                     criterion=criterion,
                     device = device)
        
        # performance evaluatons
        (_,r2_train) = test(model = model,data_loader = train_loader, criterion = criterion, device = device)
        (val_loss, r2_val) = test(model = model, data_loader=val_loader, criterion = criterion, device = device)
        (test_loss, r2_test) = test(model = model, data_loader=test_loader, criterion = criterion, device = device)
        (test_loss, r2_test) = test(model = model, data_loader=test_loader, criterion = criterion, device = device)
        
        # early stop
        if early_stop : 
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # mse
        train_loss_list.append(loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)
        
        # r2
        train_r2_list.append(r2_train)
        val_r2_list.append(r2_val)
        test_r2_list.append(r2_test)
        
        
        #save params
        if (epoch+1) % save_weights_frequency == 0: 
            torch.save(model.state_dict(), os.path.join(path,'cnn3d_epoch_'+str(epoch+1)+'.pt'))
        
        print(f'Epoch: {epoch:03d}, train loss: {loss : .4f}, val loss: {val_loss:.4f}, test loss : {test_loss:.4f}')
        
        print(f'Epoch: {epoch:03d}, train r2: {r2_train : .4f}, val r2: {r2_val:.4f}, test r2: {r2_test:.4f}')
        
       
    return model, history(metrics(r2_train, train_loss_list), metrics(r2_val, val_loss_list), metrics(r2_test, test_loss_list))


    
    

    
