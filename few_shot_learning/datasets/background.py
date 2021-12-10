import torch.utils.data as data
import torch
import numpy as np
from glob import glob
import os
import h5py
import datasets.data_gen as dg

def get_background_loader(config):
    
    background_files = [file for file in glob(os.path.join(config.experiment.train.background_path, '*.h5'))]
    
    hf = h5py.File(background_files[0])
    data_shape = hf['feat_neg'][0].shape
    hf.close()
    
    background_data = torch.zeros(0, data_shape[0], data_shape[1])
    
    for file in background_files:
        hf = h5py.File(file)
        background_data = torch.cat((background_data, torch.tensor(hf['feat_neg'][:])))
        hf.close()
    
    tr_data = dg.Datagen(config)
    background_data = tr_data.feature_scale(background_data)
    
    tensor_data = torch.utils.data.TensorDataset(background_data, torch.zeros(len(background_data), dtype=torch.long))
    background_loader = torch.utils.data.DataLoader(dataset=tensor_data, batch_sampler=BackgroundSampler(config, len(background_data)), pin_memory=True, shuffle=False, num_workers=0)
    
    return background_loader
    
class BackgroundSampler(data.Sampler):
    
    def __init__(self, config, data_size):
        self.config = config
        self.data_size = data_size
    
    def __len__(self):
        #Large number
        return int(1e100)
    
    def __iter__(self):
        
        while True:
            yield torch.unsqueeze(torch.tensor(np.random.choice(list(range(self.data_size)))), 0)
