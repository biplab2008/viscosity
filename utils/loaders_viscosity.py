import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import List, Optional
from glob import glob


# dataset class for 3DCNN
class Dataset_3DCNN(Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, 
                 path : str, 
                 folders : List[str], 
                 labels : List[float], 
                 frames : List[int], 
                 transform : Optional[transforms.Compose] = None):
        "Initialization"
        self.path = path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame_{:01d}.jpg'.format(i))).convert('L')

            if use_transform is not None:
                image = use_transform(image)
            else:
                image = transforms.ToTensor()(image)

            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.path, folder, self.transform).unsqueeze_(0)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                             # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y
        
        
        
        
# train test splitting
def create_datasets(path : str = r'D:\All_files\pys\AI_algos\Mikes_Work\viscosity-video-classification\code_digdiscovery\new_honey_164', # absolute path
                    validation_split : float = 0.2,
                    test_split : float = 0.2,
                    batch_size : int = 32,
                    transform : transforms.Compose = transforms.Compose([transforms.Resize([256, 342]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])]),
                    random_seed : int = 112,
                    shuffle : bool = True,
                    selected_frames : List[int] = [0,10,20]):
    

    all_X_list = [filename for filename in os.listdir(path)]
    all_y_list = [int(filename) for filename in os.listdir(path)]

    # train, test split
    train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=test_split, random_state=random_seed)
    
    
    
    train_set, test_set = Dataset_3DCNN(path, train_list, train_label, selected_frames, transform=transform), \
                       Dataset_3DCNN(path, test_list, test_label, selected_frames, transform=transform)
    print('length test set ', len(test_set))
   
   # split into training and validation batches
    num_train = len(train_list)
    indices = list(range(num_train))
    
    if shuffle : 
        np.random.seed(random_seed)       
        np.random.shuffle(indices)
    
    split = int(np.floor(validation_split * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)

    valid_sampler = SubsetRandomSampler(valid_idx)

    # loading train, validation and test data
    train_loader = DataLoader(train_set,
                           batch_size=batch_size,
                           sampler=train_sampler,
                           num_workers=0)
    valid_loader = DataLoader(train_set,
                           batch_size=batch_size,
                           sampler=valid_sampler,
                           num_workers=0)

    test_loader = DataLoader(test_set,
                           batch_size=batch_size,
                           num_workers=0)

    

    return train_loader, test_loader, valid_loader