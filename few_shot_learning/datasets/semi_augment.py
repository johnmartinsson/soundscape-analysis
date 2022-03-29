import torch.utils.data as data
import numpy as np
import h5py
import datasets.data_gen as dg
import torch
import torch.utils.data as data
from glob import glob
import os

def get_augment_loader(config, class_map=None, class_dict=None, tr_cls_keys=None, val_cls_keys=None):
    
    sampler = SemiAugmentSampler(config, class_map, class_dict, tr_cls_keys, val_cls_keys)
    data_set = SemiAugmentSet(config, class_map, class_dict, tr_cls_keys)
    
    loader = torch.utils.data.DataLoader(dataset=data_set, batch_sampler=sampler, pin_memory=True, shuffle=False, num_workers=0)
    
    return loader

class SemiAugmentSet(data.Dataset):
    
    def __init__(self, config, class_map=None, class_dict=None, tr_cls_keys=None):
        self.config = config
        if config.experiment.set.trainvalcv:
            self.tr_data = dg.DatagenTrainValCV(config, class_map, class_dict, tr_cls_keys)
        else:
            #For normalization
            self.tr_data = dg.Datagen(config)
    
    def __len__(self):
        return int(1e100)
    
    def __getitem__(self, idx_obj):
        
        if self.config.experiment.set.trainvalcv:
            '''
            Leave this be for now.
            Just test the regular split with this idea.
            '''
            pass
        else:
            s, h_path, i = idx_obj
            
            data = h5py.File(h_path)
            point = self.tr_data.feature_scale(torch.tensor(data['feat_pos'][i]))
            data.close()
            return point

'''
Intention: Return a batch of points from one class in the val/test sets, the points belonging to the set of n-shots given for that class.
'''
class SemiAugmentSampler(data.Sampler):
    
    def __init__(self, config, class_map=None, class_dict=None, tr_cls_keys=None, val_cls_keys=None):
        self.config = config
        self.class_map = class_map
        self.class_dict = class_dict
        self.tr_cls_keys = tr_cls_keys
        self.val_cls_keys = val_cls_keys
        
        self.structure = {}
        
        if self.config.experiment.set.trainvalcv:
            if self.config.experiment.train.semi_use_val:
                for key in val_cls_keys:
                    for inner in class_map[key]:
                        num_shots = len(class_dict[inner]['start_pos'])
                        if num_shots < config.experiment.train.n_shot + config.experiment.train.n_query:
                            continue
                        class_file_path = class_dict[inner]['file_path']
                        file_name = class_file_path.split('/')[-1]
                        h_path = os.path.join(config.experiment.path.trainvalcv_features, file_name.replace('.csv', '.h5'))
                        h5_file = h5py.File(h_path, 'r')
                        num_samples = len(h5_file[inner+'_pos_fs'][:])
                        self.structure[inner] = ('val', h_path, num_samples)
                        
            if self.config.experiment.train.semi_use_test:
                test_files = [file for file in glob(os.path.join(config.experiment.train.semi_test_path, '*.h5'))]
                i=0
                for file in test_files:
                    num_samples = len(h5py.File(file)['feat_pos'][:])
                    self.structure['t_'+str(i)] = ('test', file, num_samples)
                    i += 1
                
        else:
            all_hfiles = []
            if self.config.experiment.train.semi_use_val:
                val_files = [file for file in glob(os.path.join(config.experiment.train.semi_val_path, '*.h5'))]
                for val_file in val_files:
                    all_hfiles.append(val_file)
            if self.config.experiment.train.semi_use_test:
                test_files = [file for file in glob(os.path.join(config.experiment.train.semi_test_path, '*.h5'))]
                for test_file in test_files:
                    all_hfiles.append(test_file)
            for h_file in all_hfiles:
                samples = h5py.File(h_file, 'r')['feat_pos'][:]
                num_samples = len(samples)
                self.structure[h_file] = num_samples
    
    def __len__(self):
        return int(1e100)
    
    def __iter__(self):
        
        while True:
        
            #elemts of list tuples (set, h_file_path, idx in hdf dataset (n_shots))
            batch = []
            if self.config.experiment.set.trainvalcv:
                r_class = np.random.choice(list(self.structure.keys()))
                samples_to_take = self.config.experiment.train.n_shot + self.config.experiment.train.n_query
                struc_obj = self.structure[r_class]
                if struc_obj[0] == 'val':
                    idxs = np.random.choice(list(range(self.structure[r_class][2])))
                    for e in idxs:
                        batch.append('val', r_class, self.structure[r_class][1], e)
                else:
                    idxs = np.random.choice(list(range(self.structure[r_class][2])))
                    for e in idxs:
                        batch.append('test', None, self.structure[r_class][1], e)
            else:
                h_file_path = np.random.choice(list(self.structure.keys()))
                samples_to_take = self.config.experiment.train.n_shot + self.config.experiment.train.n_query
                idxs = np.random.choice(list(range(self.structure[h_file_path])), size=samples_to_take)
                for e in idxs:
                    batch.append((None, h_file_path, e))

            yield batch
        
        
            
            
            
            
        
        
    