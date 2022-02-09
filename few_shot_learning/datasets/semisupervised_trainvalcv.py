import sys
sys.path.append('..')

from glob import glob
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import h5py 
import librosa
import numpy as np
from sklearn.model_selection import KFold
import torch.utils.data as data
import torch
from collections import defaultdict
import datasets.data_gen as dg

class SemiSupervisedSampler_TrainValCV(data.Sampler):

    def __init__(self, config, structure, class_map, class_dict, tr_cls_keys, val_cls_keys):
        self.config = config
        self.structure = structure
        self.num_segments = config.experiment.train.n_shot + config.experiment.train.n_query
        self.class_map = class_map
        self.class_dict = class_dict
        self.tr_cls_keys = tr_cls_keys
        self.val_cls_keys = val_cls_keys
        
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
            
class SemiSupervisedSet_TrainValCV(data.Dataset):
    
    def __init__(self, config, class_map, class_dict, tr_cls_keys, val_cls_keys):
        self.config = config
        #self.tr_data = dg.Datagen(config)
        self.class_map = class_map
        self.class_dict = class_dict
        self.tr_cls_keys = tr_cls_keys
        self.val_cls_keys = val_cls_keys
        self.tr_data = dg.DatagenTrainValCV(config, class_map, class_dict, tr_cls_keys)
    
    def __len__(self):
        return int(1e100)
    
    #idx list of tuples provided by SemiSupervisedSampler
    def __getitem__(self, idx):
        s, f_path, i = idx
        
        data = h5py.File(f_path)
        if s == 'test':
            point = self.tr_data.feature_scale(torch.tensor(data['feat_neg'][i]))
        else:
            point = self.tr_data.feature_scale(torch.tensor(data['all'][i]))
        data.close()
        return point, torch.tensor(0)
            
def get_semi_structure_TrainValCV(config, class_map, class_dict, tr_cls_keys, val_cls_keys):
    
    '''
    Structure essentially is this
    keys to subdicts are the files inherent to that set.
    values are the 'length' of the files.
    
    Question: Should we try to keep them separated or not?
    For example the same file could be in both train and val here.
    I think we can try just going as previosly. If not exactly the same file
    environments were similar before and that kinda worked I suppose.
    '''
    
    semi_structure = {
        'train' : {},
        'val' : {},
        'test' : {}
    }
    
    '''
    If we just change this part we can most likely just keep the rest as is?
    
    train_files = [file for file in glob(os.path.join(config.experiment.train.semi_train_path, '*.h5'))]
    val_files = [file for file in glob(os.path.join(config.experiment.train.semi_val_path, '*.h5'))]
    test_files = [file for file in glob(os.path.join(config.experiment.train.semi_test_path, '*.h5'))]
    '''
    
    tr_tmp = []
    for key in tr_cls_keys:
        for inner in class_map[key]:
            tr_tmp.append(class_dict[inner]['file_path'])
    
    val_tmp = []
    for key in val_cls_keys:
        for inner in class_map[key]:
            val_tmp.append(class_dict[inner]['file_path'])
    
    #List comprehensions to fix paths to h5 files
    train_files = [os.path.join(config.experiment.path.trainvalcv_features, e.replace('.csv', '.h5').split('/')[-1]) for e in list(set(tr_tmp))]
    val_files = [os.path.join(config.experiment.path.trainvalcv_features, e.replace('.csv', '.h5').split('/')[-1]) for e in list(set(val_tmp))]
    test_files = [file for file in glob(os.path.join(config.experiment.train.semi_test_path, '*.h5'))]
    
    
    #TODO: Update for new feature extraction
    #The keys need to be whole paths here I think.
    for tr_file in train_files:
        hf_file = h5py.File(tr_file)
        semi_structure['train'][tr_file] = len(hf_file['all'][:])
        hf_file.close()
    
    for val_file in val_files:
        hf_file = h5py.File(val_file)
        semi_structure['val'][val_file] = len(hf_file['all'][:])
        hf_file.close()
    
    #This is pointed to from earlir extractions
    for test_file in test_files:
        hf_file = h5py.File(test_file)
        semi_structure['test'][test_file] = len(hf_file['feat_neg'][:])
        hf_file.close()
        
    return semi_structure

def get_semi_loader_TrainValCV(config, class_map, class_dict, tr_cls_keys, val_cls_keys):
    
    structure = get_semi_structure_TrainValCV(config, class_map, class_dict, tr_cls_keys, val_cls_keys)
    sampler = SemiSupervisedSampler_TrainValCV(config, structure, class_map, class_dict, tr_cls_keys, val_cls_keys)
    data_set = SemiSupervisedSet_TrainValCV(config, class_map, class_dict, tr_cls_keys, val_cls_keys)
    loader = torch.utils.data.DataLoader(dataset=data_set, batch_sampler=sampler, pin_memory=True, shuffle=False, num_workers=0)
    
    return loader
