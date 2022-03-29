import torch.utils.data as data
import torch
import numpy as np


class RandomEpisodicSampler(data.Sampler):
    
    #Include the option to choose the number of query samples
    #Y_train -> labels, just a list of the targets (list of ints?)
    def __init__(self, labels, n_episodes, n_way, n_support, n_query, config):
        
        #Number of episodes per epoch. len(labels)/(n_support * n_query) ?
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_samples = n_support+n_query
        self.config = config
        
        if config.experiment.train.print:
            self.f = open('sampler.txt', 'w')
        
        labels = np.array(labels)
        self.sample_indices = []
        for i in range(max(labels) + 1):
            ix = np.argwhere(labels == i).reshape(-1)
            ix = torch.from_numpy(ix)
            self.sample_indices.append(ix)
            
        if self.n_way > len(self.sample_indices):
            #print(self.n_way)
            raise ValueError('Error: "n_way" parameter is higher than the unique number of classes')
    
    def __len__(self):
        return self.n_episodes
    
    def __iter__(self):
        for batch in range(self.n_episodes):
            batch = []
            classes = torch.randperm(len(self.sample_indices))[:self.n_way]
            if self.config.experiment.train.print:
                self.f.write(str(classes))
            for c in classes:
                #l is a list of indexes of elements in target belonging to class c
                l = self.sample_indices[c]
                pos = torch.randperm(len(l))[:self.n_samples]
                if self.config.experiment.train.print:
                    self.f.write(str(pos))
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
        
