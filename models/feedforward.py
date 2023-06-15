from torch import nn
import torch.nn.functional as F

class LinLayers(nn.Module):
    
    def __init__(self, args):
        super(LinLayers,self).__init__()
        
        in_dim= args.in_dim #16,
        hidden_layers= args.hidden_layers #[512,256,128,2],
        activations=args.activations#[nn.LeakyReLU(0.2),nn.LeakyReLU(0.2),nn.LeakyReLU(0.2)],
        batchnorms=args.bns#[True,True,True],
        dropouts = args.dropouts#[None, 0.2, 0.2]
                
        
        assert len(hidden_layers) == len(activations) == len(batchnorms) == len(dropouts), 'dimensions mismatch!'
       
        
        layers=nn.ModuleList()
        
        if hidden_layers:
            old_dim=in_dim
            for idx,layer in enumerate(hidden_layers):
                sub_layers = nn.ModuleList()
                sub_layers.append(nn.Linear(old_dim,layer))
                if batchnorms[idx] : sub_layers.append(nn.BatchNorm1d(num_features=layer))
                if activations[idx] : sub_layers.append(activations[idx])
                if dropouts[idx] : sub_layers.append(nn.Dropout(p=dropouts[idx]))
                old_dim = layer
                
                sub_layers = nn.Sequential(*sub_layers) 
            
                layers.append(sub_layers)
            
        
            
        else:# for single layer
            layers.append(nn.Linear(in_dim,out_dim)) 
            if batchnorms : layers.append(nn.BatchNorm1d(num_features=out_dim))
            if activations : layers.append(activations)
            if dropouts : layers.append(nn.Dropout(p=dropouts))
        
        self.layers = nn.Sequential(*layers)
            
    
        
    def forward(self,x):
    
        x = self.layers(x)
        
        return x
        
    '''
    def _check_dimensions(self):
        if isinstance(self.hidden_layers,list) : 
            assert len(self.hidden_layers)==len(self.activations)
            assert len(self.hidden_layers)==len(self.batchnorms)
            assert len(self.hidden_layers)==len(self.dropouts)
    '''
        
        
        
        
