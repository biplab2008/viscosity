from typing import List, Optional, Tuple
from models.viscosity_models import CNN3D
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

@torch.no_grad()
def get_inference(model  : CNN3D, data_loader : DataLoader,  device : torch.device) -> Tuple[float, float]:
    
    y_h_all = []
    y_all =[]
    for (X,y) in data_loader:
        X = X.to(device)
        
        y = y.to(torch.float32) 
        
        y_h = model(X) 
        
        y_h_all.extend(y_h.detach().cpu().numpy())
        y_all.extend(y.numpy())
        
    df = pd.DataFrame({'y': np.array(y_all).ravel(), 'y_h': np.array(y_h_all).ravel()})

    return (df, r2_score(np.array(y_all),np.array(y_h_all)))
    
    
    
def combine_train_and_val(df_train,df_val):

    df = pd.concat([df_train,df_val])
    
    r2 = r2_score(df['y'],df['y_h'])
    
    return df,r2
    

    
 