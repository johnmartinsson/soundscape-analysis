import torch.utils.data as data
import torch
import numpy as np
from glob import glob
import os
import h5py
import datasets.data_gen as dg

'''
Class to provide unlabeled data for prototypical network training.
We also need some dataset. Probably possible to keep all in RAM for now.

Note: It is currently insanely RAM hungry :)
'''

def get_semi_loader(config):
    
    print('Creating semi supervised loader')
    
    '''
    Build dataset, somehow keep track of indices.
    We can maybe have some struct like object for this and pass it to the sampler.
    '''
    train_files = [file for file in glob(os.path.join(config.experiment.train.semi_train_path, '*.h5'))]
    val_files = [file for file in glob(os.path.join(config.experiment.train.semi_val_path, '*.h5'))]
    test_files = [file for file in glob(os.path.join(config.experiment.train.semi_test_path, '*.h5'))]
    
    hf_train = h5py.File(train_files[0])
    train_shape = hf_train['feat_neg'][0].shape
    hf_train.close()
    
    hf_val = h5py.File(val_files[0])
    val_shape = hf_val['feat_neg'][0].shape
    hf_val.close()
    
    hf_test = h5py.File(test_files[0])
    test_shape = hf_test['feat_neg'][0].shape
    hf_test.close()
    
    idx = {}
    train_data = np.empty(shape=(0,train_shape[0],train_shape[1]))
    val_data = np.empty(shape=(0,val_shape[0],val_shape[1]))
    test_data = np.empty(shape=(0,test_shape[0],test_shape[1]))
    
    #Would ultimately like to have even more information about the indexes right?
    #Like per file more or less. Disregard this at the moment. The probability of repeatedly choosing 
    #10 segments that cross the boundary of two files is abyssmal most likely
    
    #Should make it so it actually doesnt do this for sets we are not using
    '''
    TODO: Sanity check this logic.
    '''
    idx['train'] = {'start': 0, 'end': 0}
    if config.experiment.train.semi_use_train:
        for tr_file in train_files:
            hf = h5py.File(tr_file)
            train_data = np.concatenate((train_data, hf['feat_neg'][:]), axis=0)
            hf.close()
        idx['train']['end'] = len(train_data)
    
    idx['val'] = {'start': idx['train']['end'], 'end': 0}
    if config.experiment.train.semi_use_val:
        for v_file in val_files:
            hf = h5py.File(v_file)
            val_data = np.concatenate((val_data, hf['feat_neg'][:]), axis=0)
            hf.close()
        idx['val']['end'] = idx['val']['start'] + len(val_data)
    
    idx['test'] = {'start': idx['val']['end'], 'end': 0}
    if config.experiment.train.semi_use_test:
        for te_file in test_files:
            hf = h5py.File(te_file)
            test_data = np.concatenate((test_data, hf['feat_neg'][:]), axis=0)
            hf.close()
        idx['test']['end'] = idx['test']['start'] + len(test_data)
    
    sampler = SemiSupervisedSampler(config, idx)
    
    #This can be done somewhat differently to conserve memory.
    #TODO: This wont actually work if the sets have different dimensions Work on this.
    #TODO: Need to scale the data duh!
    data = torch.tensor(np.concatenate((train_data, val_data, test_data), axis=0))
    tr_data = dg.Datagen(config)
    #Scale this just as any other data I suppose.
    data = tr_data.feature_scale(data)
    data_set = torch.utils.data.TensorDataset(data, torch.zeros(len(data), dtype=torch.long))
    
    loader = torch.utils.data.DataLoader(dataset=data_set, batch_sampler=sampler, pin_memory=True, shuffle=False, num_workers=0)
    return loader

class SemiSupervisedSampler(data.Sampler):

    def __init__(self, config, idx):
        self.config = config
        self.idx = idx
        
        #could be redone but whatever
        self.idx_in_use = torch.zeros(0, dtype=int)
        if config.experiment.train.semi_use_train:
            self.idx_in_use = torch.cat((self.idx_in_use, torch.arange(idx['train']['start'], idx['train']['end'])))
        if config.experiment.train.semi_use_val:
            self.idx_in_use = torch.cat((self.idx_in_use, torch.arange(idx['val']['start'], idx['val']['end'])))
        if config.experiment.train.semi_use_test:
            self.idx_in_use = torch.cat((self.idx_in_use, torch.arange(idx['test']['start'], idx['test']['end'])))
        
        
    def __len__(self):
        #Large number
        return int(1e100)
    
    #This could just be an endless generator right? Why not. Just create one of them.
    def __iter__(self):
        
        #Yield just an array. Nothing fancy. Like peas in a pod.
        while True:
            batch = torch.zeros(0, dtype=int)
            num_segments = self.config.experiment.train.semi_n_shot + self.config.experiment.train.semi_n_query
            
            if self.config.experiment.train.semi_use_train:
                tr_i = torch.arange(self.idx['train']['start'], self.idx['train']['end'])
                for i in range(self.config.experiment.train.semi_k_way):
                    r = np.random.randint(0, len(tr_i)-num_segments)
                    batch = torch.cat((batch, tr_i[r:r+num_segments]))
                    
            if self.config.experiment.train.semi_use_val:
                val_i = torch.arange(self.idx['val']['start'], self.idx['val']['end'])
                for i in range(self.config.experiment.train.semi_k_way):
                    r = np.random.randint(0, len(val_i)-num_segments)
                    batch = torch.cat((batch, val_i[r:r+num_segments]))
            
            if self.config.experiment.train.semi_use_test:
                test_i = torch.arange(self.idx['test']['start'], self.idx['test']['end'])
                for i in range(self.config.experiment.train.semi_k_way):
                    r = np.random.randint(0, len(test_i)-num_segments)
                    batch = torch.cat((batch, test_i[r:r+num_segments]))
                    
            yield batch
                    
        
        
        
        