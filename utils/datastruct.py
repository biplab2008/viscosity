from typing import NamedTuple, List, Callable, List, Tuple, Optional
import torch

class LinData(NamedTuple):
    in_dim : int # input dimension
    hidden_layers : List[int] # hidden layers including the output layer
    activations : List[Optional[Callable[[torch.Tensor],torch.Tensor]]] # list of activations
    bns : List[bool] # list of bools
    dropouts : List[Optional[float]] # list of dropouts probas
        
class CNNData(NamedTuple):
    in_dim : int # input dimension
    n_f : List[int] # num filters
    kernel_size : List[Tuple] # kernel size [(5,5,5), (3,3,3),(3,3,3)]
    activations : List[Optional[Callable[[torch.Tensor],torch.Tensor]]] # activation list
    bns : List[bool] # batch normialization [True, True, False]
    dropouts : List[Optional[float]] # # list of dropouts probas [.5,0,0]
    #dropouts_ps : list # [0.5,.7, 0]
    paddings : List[Optional[Tuple]] #[(0,0,0),(0,0,0), (0,0,0)]
    strides : List[Optional[Tuple]] #[(1,1,1),(1,1,1),(1,1,1)]
    
    
class NetData(NamedTuple):
    cnn3d : CNNData
    lin : LinData


'''   
class LinData(NamedTuple):
    in_dim : int
    #num_classes : int
    hidden_layers : list
    activations : list
    bns : list
    dropouts : list
     
class CNNData(NamedTuple):
    in_dim : int # input dimension
    n_f : list # num filters
    kernel_size : list # kernel size [(5,5,5), (3,3,3),(3,3,3)]
    activations : list # activation list
    bns : list # batch normialization [True, True, False]
    dropouts : list # [True, True, False]
    #dropouts_ps : list # [0.5,.7, 0]
    paddings : list #[(0,0,0),(0,0,0), (0,0,0)]
    strides : list #[(1,1,1),(1,1,1),(1,1,1)]
        
class NetData(NamedTuple):
    cnn3d : CNNData
    lin : LinData
    
    
class Mdata(NamedTuple):
    cm : list
    ba : float
    sn : float
    sp : float
    tn : int
    fp : int
    fn : int
    tp : int
 
class MetricData(NamedTuple):
    train : Mdata
    test : Mdata
    
'''
    
    
# for outputs
class history():
    def __init__(self, train, val, test):
        self.train = train
        self.test = test
        self.val = val
        
class metrics():
    def __init__(self, r2, loss):
        self.r2 = r2
        self.loss = loss
   


       