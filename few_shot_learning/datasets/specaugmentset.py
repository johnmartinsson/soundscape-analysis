import torch.utils.data as data

import torch
import numpy as np

'''
This approach might not be the most desirable.
It's neat but better to augment whole batches in the trainin loop
most likely. Especially considering that we might want to augment examples
from different sets. unlabeled/labeled.
'''


class SpecAugmentSet(data.Dataset):
    
    #Adding config here for all the different specaugment options.
    def __init__(self, X, Y, config):
        self.X = X
        self.Y = Y
        self.config = config
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        pass