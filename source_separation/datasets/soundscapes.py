import glob
import os

import torch
import numpy as np
import librosa

from typing import Tuple
from torch import Tensor

def load_dataset(cfg):
    train_soundscapes = Soundscapes(
            root_dir = '../data/soundscapes/soundscape-dataset',
            split    = 'train',
            nb_samples = 10000,
            sample_rate = 22050,
            segment_length = 2,
            seed = 42)
    valid_soundscapes = Soundscapes(
            root_dir = '../data/soundscapes/soundscape-dataset',
            split    = 'valid',
            nb_samples = 1000,
            sample_rate = 22050,
            segment_length = 2,
            seed = 42)
    test_soundscapes = Soundscapes(
            root_dir = '../data/soundscapes/soundscape-dataset',
            split    = 'test',
            nb_samples = 2000,
            sample_rate = 22050,
            segment_length = 2,
            seed = 42)

    print(cfg)
    batch_size = cfg['batch_size']
    train_loader = torch.utils.data.DataLoader(train_soundscapes, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(train_soundscapes, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(train_soundscapes, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, valid_loader, test_loader

class Soundscapes(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, 
                 split: str, 
                 nb_samples: int, 
                 sample_rate: int, 
                 segment_length: int, 
                 seed: int) -> None:

        self.nb_samples = nb_samples

        # Wave file settings
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.root_dir = root_dir
        self.truncated_length = int(sample_rate * (segment_length - segment_length*0.1))
        
        # Split fractions
        self.percentage_train_split = 0.85
        self.percentage_valid_split = 0.05
        self.percentage_test_split  = 0.10
        
        # Deterministic random generator
        self.random = np.random.RandomState(seed)
        
        # Category tree
        self.category_tree = {
            'anthropophony' : ['road_traffic'],
            'geophony' : ['wind', 'rain'],
            'biophony' : ['birds', 'frogs']
        }
        
        # Initialize the split ranges for each leaf category.
        # The range of audio segments that are used for each 
        # split of the data
        self.leaf_category_range = {}
        self.leaf_category_files = {}
        for root_category in self.category_tree.keys():
            for leaf_category in self.category_tree[root_category]:
                files = glob.glob(os.path.join(self.root_dir, root_category, '{}_*.wav'.format(leaf_category)))
                self.leaf_category_files[leaf_category] = files
                
                nb_files = len(files)
                if split == 'train':
                    start_index = 0
                    end_index = int(nb_files*self.percentage_train_split)
                    self.leaf_category_range[leaf_category] = (start_index, end_index)
                elif split == 'valid':
                    start_index = int(nb_files*self.percentage_train_split)
                    end_index = int(nb_files*self.percentage_train_split) + int(nb_files*self.percentage_valid_split)
                    self.leaf_category_range[leaf_category] = (start_index, end_index)
                elif split == 'test':
                    start_index = int(nb_files*self.percentage_train_split) + int(nb_files*self.percentage_valid_split)
                    end_index = int(nb_files*self.percentage_train_split) + int(nb_files*self.percentage_valid_split) + int(nb_files*self.percentage_test_split)
                    self.leaf_category_range[leaf_category] = (start_index, end_index)
                else:
                    raise ValueError('split: {} not defined.'.format(split))
        
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]: #, int, int, int]:
        nb_root_categories = self.random.randint(1, len(self.category_tree.keys())+1)
        root_categories = self.random.choice(list(self.category_tree.keys()), nb_root_categories, replace=False)
        mix = np.zeros(self.truncated_length).astype(np.float32)
        sources = {
            'anthropophony' : np.zeros(self.sample_rate * self.segment_length).astype(np.float32),
            'geophony' : np.zeros(self.sample_rate * self.segment_length).astype(np.float32),
            'biophony' : np.zeros(self.sample_rate * self.segment_length).astype(np.float32),
        }

        for root_cetegory in root_categories:
            leaf_categories = self.category_tree[root_cetegory]
            leaf_category = self.random.choice(leaf_categories, 1)[0]
            idx = self.random.randint(*self.leaf_category_range[leaf_category])
            file = self.leaf_category_files[leaf_category][idx]
            wav, sr = librosa.load(file, sr=self.sample_rate)
            
            if len(wav) > self.truncated_length:
                wav = wav[:self.truncated_length]
            
            sources[root_cetegory] = wav
            mix += wav
            
        mix = mix/nb_root_categories
            
        return mix, sources['anthropophony'], sources['geophony'], sources['biophony']
    
    def __len__(self) -> int:
        return self.nb_samples
