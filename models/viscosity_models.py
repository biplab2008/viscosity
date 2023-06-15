import torch
from torch import nn
from utils.datastruct import NetData
from models.feedforward import LinLayers


class CNN3D_Mike(nn.Module):
    def __init__(self, t_dim=30, img_x=256 , img_y=342, drop_p=0, fc_hidden1=256, fc_hidden2=256):
        super(CNN3D_Mike, self).__init__()        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        #self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)        
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2*self.conv2_outshape[0]*self.conv2_outshape[1]*self.conv2_outshape[2],
                             self.fc_hidden1)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2,1)  # fully connected layer, output = multi-classes 
        
        
    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
       
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc3(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        #x = self.fc3(x)
        #x = F.softmax(self.fc2(x))
        
        x = self.fc3(x) 
        
        
        
        return x

    

class CNNLayers(nn.Module):

    def __init__(self, args):
    
        super(CNNLayers, self).__init__()
        
        self.in_dim = args.in_dim# 1/3
        self.n_f = args.n_f#[32,64]
        self.kernel_size = args.kernel_size # [(5,5,5), (3,3,3)]
        self.activations = args.activations#['relu', 'relu']
        self.bns = args.bns #[True, True], 
        self.dropouts = args.dropouts #[True, True]
        #self.dropouts_ps = args.dropouts_ps#[0.5,.7]
        self.paddings = args.paddings #[(0,0,0),(0,0,0)]
        self.strides = args.strides # strides [(1,1,1),(1,1,1),(1,1,1)])
        #self.poolings = args.poolings
        
        assert len(self.n_f) == len(self.activations) == len(self.bns) == len(self.dropouts), 'dimensions mismatch : check dimensions!'
        
        # generate layers seq of seq 
        self._get_layers()
       
    def _get_layers(self):
        
        layers =nn.ModuleList()
        in_channels = self.in_dim
        
        for idx, chans in enumerate(self.n_f):
            sub_layers = nn.ModuleList()                            
                                        
            sub_layers.append(nn.Conv3d(in_channels = in_channels,
                                        out_channels = chans, #self.n_f[idx],
                                        kernel_size = self.kernel_size[idx],
                                        stride = self.strides[idx],
                                        padding = self.paddings[idx]
                                        ))
                                        


            if self.bns[idx] : sub_layers.append(nn.BatchNorm3d(num_features = self.n_f[idx]))

            #if self.dropouts[idx] : sub_layers.append(nn.Dropout3d(p = self.dropouts_ps[idx]))
            
            if self.dropouts[idx] : sub_layers.append(nn.Dropout3d(p = self.dropouts[idx]))

            #if self.activations[idx]  : sub_layers.append(self.__class__.get_activation(self.activations[idx]))
            
            if self.activations[idx]  : sub_layers.append(self.activations[idx])
            
            sub_layers = nn.Sequential(*sub_layers) 
            
            layers.append(sub_layers)
            
            in_channels = self.n_f[idx]
    
        self.layers = nn.Sequential(*layers)
        
        
    @staticmethod
    def get_activation(activation):
        if activation == 'relu':
            activation=nn.ReLU()
        elif activation == 'leakyrelu':
            activation=nn.LeakyReLU(negative_slope=0.1)
        elif activation == 'selu':
            activation=nn.SELU()
        
        return activation
        
        
        
    def forward(self, x):
        
        x = self.layers(x)
        
        return x 
        
        

class CNN3D(nn.Module):

    def __init__(self, args):
        super(CNN3D,self).__init__()
        # check datatype
        if not isinstance(args, NetData):
            raise TypeError("input must be a ParserClass")
            
        self.cnn3d = CNNLayers(args.cnn3d)

        self.lin = LinLayers(args.lin)
        
        self.in_dim = args.lin.in_dim
        
        
    def forward(self, x):
        
        # cnn 3d
        x = self.cnn3d(x)
        
        x = x.view(-1, self.in_dim)
        
        # feedforward
        x = self.lin(x)
        
        return x
        
        
        