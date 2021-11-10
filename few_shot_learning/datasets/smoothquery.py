import torch.utils.data as data
import torch
import numpy as np



#Its the average of the embeddings that we want.
class SmoothQuerySet(data.Dataset):
    
    #Data to just be the data. No tuple stuff
    def __init__(self, data, smoothing):
        
        self.data = data
        self.smoothing = smoothing
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        
        sample = torch.zeros(0, self.data.shape[1], self.data.shape[2])
        for i in range(idx-self.smoothing, idx+self.smoothing+1):
            if i < 0 or i >= len(self.data):
                #Eh, this is for torch stack, idk how goot this is.
                sample = torch.cat((sample, torch.zeros(1, self.data.shape[1], self.data.shape[2])), dim=0)
            else:
                sample = torch.cat((sample, self.data[i].reshape(1, self.data.shape[1], self.data.shape[2])), dim=0)
        
        return sample


    
        