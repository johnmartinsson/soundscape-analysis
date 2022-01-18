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

'''
TODO: 13-01-22: Process gets killed when trying out segment lengths longer than 0.2s.
                Need to reduce memory footprint of the semi loaders, lazy loading essentially.
                Problem will be even more apparent when incerasing the number of mel bins.
'''


    
'''
Idea: Just create the indexes passed on to the sampler here.
      Then pass the same something with the same idx structure to the dataset.
      I mean perhaps not that complicated idk.

      I think creating a dictionary might be nice.
      semi_data = {
          train : {
              file1 : size
              file2 : size
              file3 : size
              ...
          }
          val : {
              -||-
          }
          test : {
              -||-
          }
      }

      The sampler could then randomly select files and indexes from this structure and pass that on to the dataset
      as tuples (set, filename, i) with possibility of excluding sets.
      The set and filename is then enough of a path to open it with h5py, i correspond to the position to take the num_segments at I think.
      Note: return (set1, filename1, i1) , ... , (set1, filename1, inum_segements) from sampler. Get item is supposed to only return one item i think.
'''

def get_semi_structure(config):
    
    semi_structure = {
        'train' : {},
        'val' : {},
        'test' : {}
    }
    
    
    train_files = [file for file in glob(os.path.join(config.experiment.train.semi_train_path, '*.h5'))]
    val_files = [file for file in glob(os.path.join(config.experiment.train.semi_val_path, '*.h5'))]
    test_files = [file for file in glob(os.path.join(config.experiment.train.semi_test_path, '*.h5'))]
    
    for tr_file in train_files:
        hf_file = h5py.File(tr_file)
        semi_structure['train'][tr_file] = len(hf_file['feat_neg'][:])
        hf_file.close()
    
    for val_file in val_files:
        hf_file = h5py.File(val_file)
        semi_structure['val'][val_file] = len(hf_file['feat_neg'][:])
        hf_file.close()
        
    for test_file in test_files:
        hf_file = h5py.File(test_file)
        semi_structure['test'][test_file] = len(hf_file['feat_neg'][:])
        hf_file.close()
        
    return semi_structure

def get_semi_loader(config):
    
    
    print('Creating semi supervised loader')
    
    semi_structure = get_semi_structure(config)
    
    sampler = SemiSupervisedSampler(config, semi_structure)
    data_set = SemiSupervisedSet(config)
    
    loader = torch.utils.data.DataLoader(dataset=data_set, batch_sampler=sampler, pin_memory=True, shuffle=False, num_workers=0)
    return loader

class SemiSupervisedSet(data.Dataset):
    
    def __init__(self, config):
        self.config = config
        self.tr_data = dg.Datagen(config)
    
    def __len__(self):
        return int(1e100)
    
    #idx list of tuples provided by SemiSupervisedSampler
    def __getitem__(self, idx):
        s, f_path, i = idx
        
        data = h5py.File(f_path)
        point = self.tr_data.feature_scale(torch.tensor(data['feat_neg'][i]))
        data.close()
        return point, torch.tensor(0)

class SemiSupervisedSampler(data.Sampler):

    def __init__(self, config, structure):
        self.config = config
        self.structure = structure
        self.num_segments = config.experiment.train.n_shot + config.experiment.train.n_query
        
    def __len__(self):
        #Large number
        return int(1e100)
    
    #This could just be an endless generator right? Why not. Just create one of them.
    def __iter__(self):
        
        while True:
            
            #Tuples (set, file, i)
            batch = []
            
            if self.config.experiment.train.semi_use_train:
                train_file_paths = list(self.structure['train'].keys())
                for i in range(self.config.experiment.train.semi_k_way):
                    file_path = np.random.choice(train_file_paths)
                    file_len = self.structure['train'][file_path]
                    i = np.random.choice(np.arange(file_len-self.num_segments))
                    for j in range(self.num_segments):
                        batch.append(('train', file_path, i+j))
                    
                    
            if self.config.experiment.train.semi_use_val:
                val_file_paths = list(self.structure['val'].keys())
                for i in range(self.config.experiment.train.semi_k_way):
                    file_path = np.random.choice(val_file_paths)
                    file_len = self.structure['val'][file_path]
                    i = np.random.choice(np.arange(file_len-self.num_segments))
                    for j in range(self.num_segments):
                        batch.append(('val', file_path, i+j))
                    
            if self.config.experiment.train.semi_use_test:
                test_file_paths = list(self.structure['test'].keys())
                for i in range(self.config.experiment.train.semi_k_way):
                    file_path = np.random.choice(test_file_paths)
                    file_len = self.structure['test'][file_path]
                    i = np.random.choice(np.arange(file_len-self.num_segments))
                    for j in range(self.num_segments):
                        batch.append(('test', file_path, i+j))
                
            yield batch
                    
        
        
        
        