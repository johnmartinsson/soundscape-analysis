#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import abc
import csv
import yaml
import h5py
import librosa
import os
import hydra
from hydra import compose, initialize
from glob import glob
from itertools import chain
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


# In[2]:


'''
As of now most of the code in this notebook is more or less copied from the DCASE repository.
Minor changes have been done and more is coming to accomodate more flexible FS learning such as
active episodic training and most likely more stuff.
'''


# In[3]:


'''
How to make the framework flexible enough that one can point to which samples in a batch are meant to be
support/query per class? The implementation in DCASE2021 does not handle this.

Currently return the pcen transposed. Where to transpose it back?
Batcher? Most important thing is just to not forget i think.
This doesnt really matter as of now since the model dont care. (Time insensitive)

The code only allows one positive class per segment for now I think.
This might be something we would like to fix? (How?)
Binary applications not uninterestig though
'''


# ## Prototypical net

# In[4]:


#DCASE2021

def conv_block(in_channels,out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


# In[5]:


#DCASE2021

#TODO introduce parametrization of conv blocks?
class Protonet(nn.Module):
    def __init__(self, raw_transformer=None):
        super(Protonet,self).__init__()
        self.raw_transformer = raw_transformer
        self.encoder = nn.Sequential(
            conv_block(1,128),
            conv_block(128,128),
            conv_block(128,128),
            conv_block(128,128)
        )
    def forward(self,x):
        #Is there risk for this to be super slow?
        #A naive approach might transform the same data more than once?
        #Lookup tables?
        if self.raw_transformer is not None:
            x = self.raw_transformer.rtoi_standard(x)
        (num_samples,seq_len,mel_bins) = x.shape
        x = x.view(-1,1,seq_len,mel_bins)
        x = self.encoder(x)
        return x.view(x.size(0),-1)


# In[6]:


'''
Will most likely lean heavily on the implementation of the DCASE2021 task 5 baseline implementation.

'''
def prototypical_loss(input, target, n_support, supp_idxs=None):
    
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = target.eq(classes[0].item()).sum().item() - n_support
    if supp_idxs is None:
        #Rewrite, need to select only n_support. We might have n_query > n_support
        supp_idxs = list(map(lambda c: target_cpu.eq(c).nonzero()[:n_support].squeeze(1), classes))
        q_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    else:
        #Work from supp_idxs.
        q_idxs = None
        
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in supp_idxs])
    query_samples = input_cpu[q_idxs]
    #I think prototypes has the wrong dimension here?
    #Query samples shape (10,1024)
    #Prototypes (2,1,1024)
    dists = euclidean_dist(query_samples, prototypes)
    
    #Check
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    #.mean() -> 1/NcNq
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val
    


# ## Data processing

# In[7]:


'''
    * Design choice: Handle most of pre-processing as part of the model (torchlibrosa)?
      May ultimately lead to simpler augmentation etc down the line. Work with raw audio as far as possible?
      
    * Make use of h5py library for storing training, validation and test sets?
      Still raw audio sets?
    
    * Incorporate pytorch Dataloader, seems prudent and a good design choice.
      read(h5py) file + Episodic sampler -> Dataloader?
      
    * Slight change of mind. Datagen and FeatureExtractor is not really worth spending time on for now.
      Sure they could be interfaces for a framework up the road but can do without for now since the loop
      will most likely be quite task dependent for now.
      
'''


# In[8]:


'''
Possibly take a h5 file as input and return X_train, Y_train, X_val, Y_val
Is this an approach that we like? Is it commonly used for deep learning?
'''

#DCASE

class Datagen():
    
    def __init__(self, config):
        
        self.config = config
        
        if config.features.raw:
            #These obviosly requires more processing down the pipe but that is application dependent.
            #Leave be for now
            hf = h5py.File(os.path.join(config.path.train_w, 'raw_train.h5'))
        else:
            hf = h5py.File(os.path.join(config.path.train_w, 'mel_train.h5'))
            self.x = hf['features'][:]
            self.labels = [s.decode() for s in hf['labels'][:]]
            if config.datagen.ltoi:
                self.y = class_to_int(self.labels)
            else:
                self.y = self.labels
            if config.datagen.balance:
                self.x, self.y = balance_class_distribution(self.x, self.y)
            
            array_train = np.arange(len(self.x))
            if config.datagen.stratify:
                _,_,_,_,train_array,valid_array = train_test_split(self.x, self.y, array_train,                                                     random_state=config.datagen.random_state, stratify=self.y)
            else:
                _,_,_,_,train_array,valid_array = train_test_split(self.x, self.y, array_train,                                                     random_state=config.datagen.random_state)
                
            self.train_index = train_array
            self.valid_index = valid_array
            if config.datagen.normalize:
                self.mean, self.std = norm_params(self.x[train_array])
            else:
                self.mean = None
                self.std = None
                
    def feature_scale(self, x):
        return (x - self.mean)/self.std
    
    def generate_train(self):
        train_array = sorted(self.train_index)
        valid_array = sorted(self.valid_index)
        X_train = self.x[train_array]
        Y_train = self.y[train_array]
        X_val = self.x[valid_array]
        Y_val = self.y[valid_array]
        if self.config.datagen.normalize:
            X_train = self.feature_scale(X_train)
            X_val = self.feature_scale(X_val)
        return X_train, Y_train, X_val, Y_val
        

#In comparison to parent class instances will work on one particular hfile
#and return the relevant datasets, pos, neg, query
class TestDatagen(Datagen):
    
    def __init__(self, hfile, config):
        
        #Debatable if this should be rewritten in the case where we do not normalize.
        #Should really give this some thought overall actually?
        #Isnt this normalization somewhat weird?
        super().__init__(config)
        
        self.hfile = hfile
        
    def generate_eval(self):
        
        X_pos = self.hfile['feat_pos'][:]
        X_neg = self.hfile['feat_neg'][:]
        X_query = self.hfile['feat_query'][:]
        if self.config.datagen.normalize:
            X_pos = self.feature_scale(X_pos)
            X_neg = self.feature_scale(X_neg)
            X_query = self.feature_scale(X_query)
            
        return X_pos, X_neg, X_query
        
        


# In[9]:


'''
This could be an interface / abstract class to build audio 
to some other format instance to plug into feature extractor
'''

class Spectralizer():
    
    def __init__(self, config):
        self.config = config
        
        self.sr = config.features.sr
        self.n_fft = config.features.n_fft
        self.hop = config.features.hop_mel
        self.n_mels = config.features.n_mels
        self.fmax = config.features.fmax
        

    def raw_to_spec(self, audio, config):

        #Supposedly suggested by librosa.
        audio = audio * (2**32)

        mel_spec = librosa.feature.melspectrogram(audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop,
                                                 n_mels=self.n_mels, fmax=self.fmax)

        pcen = librosa.core.pcen(mel_spec, sr=self.sr)
        pcen = pcen.astype(np.float32)
        
        #Note that we transform the features here and therefor have time/frame along dim 0.
        #Transform back when loading data? Smaksak
        return pcen.T
    


# In[10]:


'''
Possibly work on an raw files and annotations and return/write h5 files.
This might be clunky to include in a framework since this most likely is dataset dependent.
Might however benfit from having an interface which is inherited by classes working on specific datasets.
'''

class FeatureExtractor(abc.ABC):
    
    def __init__(self):
        pass

'''
Takes the data from the DCASE (all files one folder) and returns h5 file with the datasets 'features' and 'labels'
This takes no heed to unlabeled segments and therefor we will have no unlabeled data to work with.
This is an interesting TODO. Most likely need to rework some of the mechanisms here to work with limited RAM.
Extract segment -> write to file etc... Look at DCASE code for example
Unlabeled data could be saved to a new dataset 'unlabeled' for example.


TODO: MemError already present even before processing unlabeled data and only one of the smaller audio files.
Atleast for the non raw data. Need to fix this. Probably not hard for data processed into spectrograms since
we beforehand know the dimensions. Harder for raw audio segments.

Why are we getting MemError though? Could run the DCASE program from home with 16GB RAM.
Does not load all features into memory at once? Wonky h5py thing? Check this out!

It seems the DCASE code loads all the features into memory.

Found a bug, this however does not nessecarily discard the above comments.
Working well with memory is still most likely of importance when extracting from large sets.
'''
class MyF_Ext(FeatureExtractor):
    
    def __init__(self, config, spectralizer=None):
        self.config = config
        self.spectralizer = spectralizer
        
    def extract_features(self):
        
        self.extract_train()
        self.extract_test()
    
    '''
    Assumes all *.csv and *.wav files are in the same folder which path is in config.
    Either creates spectrograms as features or raw audio segments containing events.
    Assumes annotations as those provided in 
    '''
    
    def extract_train(self):
        
        print('--- Processing training data ---')
        csv_files = [file for file in glob(os.path.join(self.config.path.data_train, '*.csv'))]
        
        if self.config.features.raw:
            
            print('Raw extraction')
            
            events = []
            labels = []
            
            for file in csv_files:
            
                print('Processing ' + file.replace('csv', 'wav'))
                audio, sr = librosa.load(file.replace('csv', 'wav'), self.config.features.sr)
                df = pd.read_csv(file, header=0, index_col=False)
                df_pos = df[(df == 'POS').any(axis=1)]
                
                #Add config options for window size around event
                df_pos.loc[:, 'Starttime'] = df_pos['Starttime'] - 0.025
                df_pos.loc[:, 'Endtime'] = df_pos['Endtime'] + 0.025
                start_time = [int(np.floor(start * sr)) for start in df_pos['Starttime']]
                end_time = [int(np.floor(end * sr)) for end in df_pos['Endtime']]
                
                #Better way of doing this?
                for i in range(len(start_time)):
                    events += [audio[start_time[i]:end_time[i]]]
                    
                labels += list(chain.from_iterable(
                    [df_pos.columns[(df_pos == 'POS').loc[index]].values for index, _ in df_pos.iterrows()]))
            
            print('Padding')
            #Pad arrays in events and format for write
            max_len = 0
            for e in events:
                if len(e) > max_len:
                    max_len = len(e)
                    
            for i in range(len(events)):
                if len(events[i]) < max_len:
                    events[i] = np.append(events[i], np.array([self.config.features.raw_pad]*(max_len-len(events[i]))))
            
            events = np.array(events)
            
            print('Writing to file')
            
            hf = h5py.File(os.path.join(self.config.path.train_w, 'raw_train.h5'), 'w')
            hf.create_dataset('features', data=events)
            hf.create_dataset('labels', data=[s.encode() for s in labels], dtype='S20')
            hf.close()
            
            print('Done')
            
        else:
            
            #DCASE more or less
            
            print('Spectrogram extraction')
            
            fps = self.config.features.sr / self.config.features.hop_mel
            seg_len = int(round(self.config.features.seg_len * fps))
            hop_seg = int(round(self.config.features.hop_seg * fps))
            
            labels = []
            events = []
            
            for file in csv_files:
                
                print('Processing ' + file.replace('csv', 'wav'))
                audio, sr = librosa.load(file.replace('csv', 'wav'), self.config.features.sr)
                
                print('Spectral transform')
                pcen = self.spectralizer.raw_to_spec(audio, self.config)
                
                df = pd.read_csv(file, header=0, index_col=False)
                df_pos = df[(df == 'POS').any(axis=1)]
                
                start_time, end_time = time_2_frame(df_pos, fps)
                label_f = list(chain.from_iterable(
                    [df_pos.columns[(df_pos == 'POS').loc[index]].values for index, _ in df_pos.iterrows()]))
                
                print('Slicing spectrogram')
                
                for index in range(len(start_time)):
                    
                    str_ind = start_time[index]
                    end_ind = end_time[index]
                    label = label_f[index]
                    
                    #Event longer than a segment?
                    if end_ind - str_ind > seg_len:
                        shift = 0
                        while end_ind - (str_ind + shift) > seg_len:
                            
                            pcen_patch = pcen[int(str_ind + shift):int(str_ind + shift + seg_len)]
                            events += [pcen_patch]
                            labels.append(label)
                            shift += hop_seg
                        
                        pcen_patch = pcen[end_ind - seg_len:end_ind]
                        events += [pcen_patch]
                        labels.append(label)
                    
                    #Event shorter than a segment!
                    else:
                        
                        #Repeat the patch til segment length.
                        pcen_patch = pcen[str_ind:end_ind]
                        if pcen_patch.shape[0] == 0:
                            continue
                        
                        repeats = int(seg_len/(pcen_patch.shape[0])) + 1
                        pcen_patch_new = np.tile(pcen_patch, (repeats, 1))
                        pcen_patch_new = pcen_patch_new[0:int(seg_len)]
                        events += [pcen_patch_new]
                        labels.append(label)
                        
            print('Writing to file')
            
            events = np.array(events)
            
            hf = h5py.File(os.path.join(self.config.path.train_w, 'mel_train.h5'), 'w')
            hf.create_dataset('features', data=events)
            hf.create_dataset('labels', data=[s.encode() for s in labels], dtype='S20')
            hf.close()
            
            print('Done')
                        
                
                        
                
    #Try to start out in a way that would make it easier to possibly incorporate multiple negative classes
    #down the line. This needs to be reflected in TestDatagen. Perhaps just list of lists with indexes to 
    #the h5 dataset 'feat_neg'. Then if this is not in keys just assume that only one negative class exists.
    
    #For now just a copy of the DCASE code, this since the way they work here most likely have an impact on the
    #scoring/evaluation metrics on the github.
    def extract_test(self):
        
        print('--- Processing test data ---')
        csv_files = [file for file in glob(os.path.join(self.config.path.data_test, '*.csv'))]
        
        #Are we ever interested in a raw extraction here?
        if self.config.features.raw:
            pass
        else:
            
            fps = self.config.features.sr / self.config.features.hop_mel
            seg_len = int(round(self.config.features.seg_len * fps))
            hop_seg = int(round(self.config.features.hop_seg * fps))
            
            for file in csv_files:
                
                print('Processing ' + file.replace('csv', 'wav'))
                
                idx_pos = 0
                idx_neg = 0
                start_neg = 0
                hop_neg = 0
                idx_query = 0
                hop_query = 0
                strt_index = 0

                split_list = file.split('/')
                name = str(split_list[-1].split('.')[0])
                feat_name = name + '.h5'
                audio_path = file.replace('csv', 'wav')
                feat_info = []
                hdf_eval = os.path.join(self.config.path.test_w ,feat_name)
                hf = h5py.File(hdf_eval,'w')
                hf.create_dataset('feat_pos', shape=(0, seg_len, self.config.features.n_mels),
                                  maxshape= (None, seg_len, self.config.features.n_mels))
                hf.create_dataset('feat_query',shape=(0,seg_len, self.config.features.n_mels),maxshape=(None,seg_len,self.config.features.n_mels))
                hf.create_dataset('feat_neg',shape=(0,seg_len, self.config.features.n_mels),maxshape=(None,seg_len,self.config.features.n_mels))
                hf.create_dataset('start_index_query',shape=(1,),maxshape=(None))

                'In case you want to use the statistics of each file to normalize'

                hf.create_dataset('mean_global',shape=(1,), maxshape=(None))
                hf.create_dataset('std_dev_global',shape=(1,), maxshape=(None))

                df_eval = pd.read_csv(file, header=0, index_col=False)
                Q_list = df_eval['Q'].to_numpy()

                start_time,end_time = time_2_frame(df_eval,fps)

                index_sup = np.where(Q_list == 'POS')[0][:self.config.train.n_shot]

                audio, sr = librosa.load(file.replace('csv', 'wav'), self.config.features.sr)
                print('Spectral transform')
                pcen = self.spectralizer.raw_to_spec(audio, self.config)
               
                mean = np.mean(pcen)
                std = np.mean(pcen)
                hf['mean_global'][:] = mean
                hf['std_dev_global'][:] = std

                strt_indx_query = end_time[index_sup[-1]]
                end_idx_neg = pcen.shape[0] - 1
                hf['start_index_query'][:] = strt_indx_query

                print("Creating negative dataset")

                while end_idx_neg - (strt_index + hop_neg) > seg_len:

                    patch_neg = pcen[int(strt_index + hop_neg):int(strt_index + hop_neg + seg_len)]

                    hf['feat_neg'].resize((idx_neg + 1, patch_neg.shape[0], patch_neg.shape[1]))
                    hf['feat_neg'][idx_neg] = patch_neg
                    idx_neg += 1
                    hop_neg += hop_seg

                last_patch = pcen[end_idx_neg - seg_len:end_idx_neg]
                hf['feat_neg'].resize((idx_neg + 1, last_patch.shape[0], last_patch.shape[1]))
                hf['feat_neg'][idx_neg] = last_patch

                print("Creating Positive dataset")
                for index in index_sup:

                    str_ind = int(start_time[index])
                    end_ind = int(end_time[index])

                    if end_ind - str_ind > seg_len:

                        shift = 0
                        while end_ind - (str_ind + shift) > seg_len:

                            patch_pos = pcen[int(str_ind + shift):int(str_ind + shift + seg_len)]

                            hf['feat_pos'].resize((idx_pos + 1, patch_pos.shape[0], patch_pos.shape[1]))
                            hf['feat_pos'][idx_pos] = patch_pos
                            idx_pos += 1
                            shift += hop_seg
                        last_patch_pos = pcen[end_ind - seg_len:end_ind]
                        hf['feat_pos'].resize((idx_pos + 1, patch_pos.shape[0], patch_pos.shape[1]))
                        hf['feat_pos'][idx_pos] = last_patch_pos
                        idx_pos += 1

                    else:
                        patch_pos = pcen[str_ind:end_ind]

                        if patch_pos.shape[0] == 0:
                            print(patch_pos.shape[0])
                            print("The patch is of 0 length")
                            continue
                        repeat_num = int(seg_len / (patch_pos.shape[0])) + 1

                        patch_new = np.tile(patch_pos, (repeat_num, 1))
                        patch_new = patch_new[0:int(seg_len)]
                        hf['feat_pos'].resize((idx_pos + 1, patch_new.shape[0], patch_new.shape[1]))
                        hf['feat_pos'][idx_pos] = patch_new
                        idx_pos += 1



                print("Creating query dataset")

                while end_idx_neg - (strt_indx_query + hop_query) > seg_len:

                    patch_query = pcen[int(strt_indx_query + hop_query):int(strt_indx_query + hop_query + seg_len)]
                    hf['feat_query'].resize((idx_query + 1, patch_query.shape[0], patch_query.shape[1]))
                    hf['feat_query'][idx_query] = patch_query
                    idx_query += 1
                    hop_query += hop_seg


                last_patch_query = pcen[end_idx_neg - seg_len:end_idx_neg]

                hf['feat_query'].resize((idx_query + 1, last_patch_query.shape[0], last_patch_query.shape[1]))
                hf['feat_query'][idx_query] = last_patch_query

                hf.close()

            
                
            
            
            
            
            


# In[11]:


#Instance with torchlibrosa to be included in model if input is raw.
#Having seconds thaughts on putting the data raw into the models.
#Extracting raw features and DataGenning them still of interest.
#But transform dataset before training?

class RawTransformer:
    
    def __init__(self, config):
        #Mel stuff etc
        self.config = config
    
    #Input is a training batch?
    def rtoi_standard(input):
        pass


# ## Episodic constructor

# In[12]:


#DCASE 2021 ish
#Instance given to DataLoader on argument batch_sampler

class RandomEpisodicSampler(data.Sampler):
    
    #Include the option to choose the number of query samples
    #Y_train -> labels, just a list of the targets (list of ints?)
    def __init__(self, labels, n_episodes, n_way, n_support, n_query):
        
        #Number of episodes per epoch. len(labels)/(n_support * n_query) ?
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_samples = n_support+n_query
        
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
            for c in classes:
                #l is a list of indexes of elements in target belonging to class c
                l = self.sample_indices[c]
                pos = torch.randperm(len(l))[:self.n_samples]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
        


# In[13]:


#Must somehow have access to all the data (just pass it).

class ActiveEpisodicSampler(data.Sampler):
    
    def __init__(self):
        pass


# ## Util/Functionality

# In[14]:


#DCASE

def time_2_frame(df,fps):


    #Margin of 25 ms around the onset and offsets
    #TODO: Should be in config

    df.loc[:,'Starttime'] = df['Starttime'] - 0.025
    df.loc[:,'Endtime'] = df['Endtime'] + 0.025

    #Converting time to frames

    start_time = [int(np.floor(start * fps)) for start in df['Starttime']]

    end_time = [int(np.floor(end * fps)) for end in df['Endtime']]

    return start_time,end_time


# In[15]:


#DCASE

def class_to_int(labels):
    
    class_set = set(labels)
    ltoix = {label:index for index, label in enumerate(class_set)}
    return np.array([ltoix[label] for label in labels])


# In[16]:


#DCASE

#Check over this
def balance_class_distribution(X,Y):

    '''  Class balancing through Random oversampling
    Args:
    -X: Feature
    -Y: labels

    Out:
    -X_new: Feature after oversampling
    -Y_new: Oversampled label list
    '''

    x_index = [[index] for index in range(len(X))]
    set_y = set(Y)


    ros = RandomOverSampler(random_state=42)
    x_unifm, y_unifm = ros.fit_resample(x_index, Y)
    unifm_index = [index_new[0] for index_new in x_unifm]

    X_new = np.array([X[index] for index in unifm_index])

    sampled_index = [idx[0] for idx in x_unifm]
    Y_new = np.array([Y[idx] for idx in sampled_index])

    return X_new,Y_new


# In[17]:


#DCASE

def norm_params(X):

    '''  Normalize features
        Args:
        - X : Features

        Out:
        - mean : Mean of the feature set
        - std: Standard deviation of the feature set
        '''


    mean = np.mean(X)

    std = np.std(X)
    return mean, std


# In[18]:


#DCASE

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        #we are currently getting stuck here?
        #why?
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)

    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


# In[19]:


#DCASE

def balance_class_distribution(X,Y):

    '''  Class balancing through Random oversampling
    Args:
    -X: Feature
    -Y: labels

    Out:
    -X_new: Feature after oversampling
    -Y_new: Oversampled label list
    '''

    x_index = [[index] for index in range(len(X))]
    set_y = set(Y)


    ros = RandomOverSampler(random_state=42)
    x_unifm, y_unifm = ros.fit_resample(x_index, Y)
    unifm_index = [index_new[0] for index_new in x_unifm]

    X_new = np.array([X[index] for index in unifm_index])

    sampled_index = [idx[0] for idx in x_unifm]
    Y_new = np.array([Y[idx] for idx in sampled_index])

    return X_new,Y_new


# In[20]:


#DCASE

#TODO Make it possible to use all samples as negatives.
#TODO Read up on this function, understand it better.


def evaluate_prototypes(conf=None,hdf_eval=None,device= None,strt_index_query=None):

    """ Run the evaluation
    Args:
     - conf: config object
     - hdf_eval: Features from the audio file
     - device:  cuda/cpu
     - str_index_query : start frame of the query set w.r.t to the original file

     Out:
     - onset: Onset array predicted by the model
     - offset: Offset array predicted by the model
      """
    hop_seg = int(conf.features.hop_seg * conf.features.sr // conf.features.hop_mel)

    gen_eval = TestDatagen(hdf_eval,conf)
    X_pos, X_neg,X_query = gen_eval.generate_eval()

    X_pos = torch.tensor(X_pos)
    Y_pos = torch.LongTensor(np.zeros(X_pos.shape[0]))
    X_neg = torch.tensor(X_neg)
    Y_neg = torch.LongTensor(np.zeros(X_neg.shape[0]))
    X_query = torch.tensor(X_query)
    Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

    num_batch_query = len(Y_query) // conf.eval.query_batch_size

    query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
    q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,batch_size=conf.eval.query_batch_size,shuffle=False)
    query_set_feat = torch.zeros(0,1024).cpu()


    Model = Protonet()

    if device == 'cpu':
        Model.load_state_dict(torch.load(conf.path.best_model, map_location=torch.device('cpu')))
    else:
        Model.load_state_dict(torch.load(conf.path.best_model))

    Model.to(device)
    Model.eval()

    'List for storing the combined probability across all iterations'
    prob_comb = []

    iterations = conf.eval.iterations
    for i in range(iterations):
        prob_pos_iter = []
        neg_indices = torch.randperm(len(X_neg))[:conf.eval.samples_neg]
        X_neg = X_neg[neg_indices]
        Y_neg = Y_neg[neg_indices]
        batch_size_neg = conf.eval.negative_set_batch_size
        neg_dataset = torch.utils.data.TensorDataset(X_neg, Y_neg)
        negative_loader = torch.utils.data.DataLoader(dataset=neg_dataset, batch_sampler=None, batch_size=batch_size_neg)

        batch_samplr_pos = RandomEpisodicSampler(Y_pos, num_batch_query + 1, 1, conf.train.n_shot, conf.train.n_query)
        pos_dataset = torch.utils.data.TensorDataset(X_pos, Y_pos)
        pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_sampler=batch_samplr_pos)

        neg_iterator = iter(negative_loader)
        pos_iterator = iter(pos_loader)
        q_iterator = iter(q_loader)

        print("Iteration number {}".format(i))

        for batch in tqdm(neg_iterator):
            x_neg, y_neg = batch
            x_neg = x_neg.to(device)
            feat_neg = Model(x_neg)
            feat_neg = feat_neg.detach().cpu()
            query_set_feat = torch.cat((query_set_feat, feat_neg), dim=0)
        neg_proto = query_set_feat.mean(dim=0)
        neg_proto =neg_proto.to(device)

        for batch in tqdm(q_iterator):
            x_q, y_q = batch
            x_q = x_q.to(device)
            #Why even bother with a data loader for the positive class?
            #Are we not only drawing the 5 sample that there is repeatedly?
            #Could just run the positives through the network once and save the
            #Prototype. Check that I am right about this.
            x_pos, y_pos = next(pos_iterator)
            x_pos = x_pos.to(device)
            x_pos = Model(x_pos)
            x_query = Model(x_q)
            probability_pos = get_probability(x_pos, neg_proto, x_query)
            prob_pos_iter.extend(probability_pos)

        prob_comb.append(prob_pos_iter)

    prob_final = np.mean(np.array(prob_comb),axis=0)

    krn = np.array([1, -1])
    prob_thresh = np.where(prob_final > conf.eval.p_thresh, 1, 0)

    prob_pos_final = prob_final * prob_thresh
    changes = np.convolve(krn, prob_thresh)

    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_time_query = strt_index_query * conf.features.hop_mel / conf.features.sr

    onset = (onset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    onset = onset + str_time_query

    offset = (offset_frames + 1) * (hop_seg) * conf.features.hop_mel / conf.features.sr
    offset = offset + str_time_query

    assert len(onset) == len(offset)
    return onset, offset


# In[21]:


#DCASE

def get_probability(x_pos,neg_proto,query_set_out):


    """Calculates the  probability of each query point belonging to either the positive or negative class
     Args:
     - x_pos : Model output for the positive class
     - neg_proto : Negative class prototype calculated from randomly chosed 100 segments across the audio file
     - query_set_out:  Model output for the first 8 samples of the query set

     Out:
     - Probabiility array for the positive class
     """

    pos_prototype = x_pos.mean(0)
    prototypes = torch.stack([pos_prototype,neg_proto])
    dists = euclidean_dist(query_set_out,prototypes)
    '''  Taking inverse distance for converting distance to probabilities'''
    inverse_dist = torch.div(1.0, dists)
    prob = torch.softmax(inverse_dist,dim=1)
    '''  Probability array for positive class'''
    prob_pos = prob[:,0]

    return prob_pos.detach().cpu().tolist()


# ## Eval stuff
# 

# In[56]:


#DCASE pretty much

#Think about how to actually choose examples of the positive class in practice?
#Can one even do this? Read paper again

def dummy_choice(csv, n_shots):
    events = []
    for i in range(len(csv)):
                if(csv.loc[i].values[-1] == 'POS' and len(events) < n_shots):
                    events.append(csv.loc[i].values)
    return events

#Might wanna check the paths here and if we are please with the output.
def post_processing(val_path, evaluation_file, new_evaluation_file, n_shots=5):
    '''Post processing of a prediction file by removing all events that have shorter duration
    than 60% of the minimum duration of the shots for that audio file.
    
    Parameters
    ----------
    val_path: path to validation set folder containing subfolders with wav audio files and csv annotations
    evaluation_file: .csv file of predictions to be processed
    new_evaluation_file: .csv file to be saved with predictions after post processing
    n_shots: number of available shots
    '''
    
    '''
    I think it is of great interest to not just choose the first five positives in practice.
    Sure this is part of the challenge. But... Interesting to invesigate. Discussion about growing
    number of supports can fit here to?
    '''
    
    csv_files = [file for file in glob(os.path.join(val_path, '*.csv'))]
    
    dict_duration = {}
    for csv_file in csv_files:
        audiofile = csv_file.replace('.csv', '.wav')
        df = pd.read_csv(csv_file)
        events = dummy_choice(df, n_shots)
        min_duration = 10000 #configurable?
        for event in events:
            if float(event[2])-float(event[1]) < min_duration:
                min_duration = float(event[2])-float(event[1])
        #dict_duration[audiofile] = min_duration
        dict_duration[os.path.split(audiofile)[1]] = min_duration

    results = []
    with open(evaluation_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            results.append(row)

    new_results = [['Audiofilename', 'Starttime', 'Endtime']]
    for event in results:
        audiofile = os.path.split(event[0])[1]
        min_dur = dict_duration[audiofile]
        if float(event[2])-float(event[1]) >= 0.6*min_dur:
            new_results.append([os.path.split(event[0])[1], event[1], event[2]])

    with open(new_evaluation_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_results)


# ## Scoring

# In[ ]:


#Look at scoring files from dcase and try to understand them.


# ## Loop

# In[23]:


#DCASE


def train(model, train_loader, val_loader, config, num_batches_tr, num_batches_val):
    
    if config.set.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    #Should this be done here or passed into this function?
    #Could be configs for more terminal flexibility
    optim = torch.optim.Adam(model.parameters(), lr=config.train.lr_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=config.train.scheduler_gamma,
                                                  step_size=config.train.scheduler_step_size)
    num_epochs = config.train.epochs
    
    best_model_path = config.path.best_model
    last_model_path = config.path.last_model
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_acc = 0.0
    model.to(device)
    
    for epoch in range(num_epochs):
        
        print('Epoch {}'.format(epoch))
        train_iterator = iter(train_loader)
        for batch in tqdm(train_iterator):
            optim.zero_grad()
            model.train()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x_out = model(x)
            tr_loss, tr_acc = prototypical_loss(x_out, y, config.train.n_shot)
            train_loss.append(tr_loss.item())
            train_acc.append(tr_acc.item())
            
            tr_loss.backward()
            optim.step()
            
        avg_loss_tr = np.mean(train_loss[-num_batches_tr:])
        avg_acc_tr = np.mean(train_acc[-num_batches_tr:])
        print('Average train loss: {}  Average training accuracy: {}'.format(avg_loss_tr,avg_acc_tr))
        
        lr_scheduler.step()
        
        #No dropouts in model for now, I think there is no difference between train and eval mode
        model.eval()
        val_iterator = iter(val_loader)
        for batch in tqdm(val_iterator):
            x, y = batch
            x = x.to(device)
            x_val = model(x)
            valid_loss, valid_acc = prototypical_loss(x_val, y, config.train.n_shot)
            val_loss.append(valid_loss.item())
            val_acc.append(valid_acc.item())
        avg_loss_val = np.mean(val_loss[-num_batches_val:])
        avg_acc_val = np.mean(val_acc[-num_batches_val:])
        
        print ('Epoch {}, Validation loss {:.4f}, Validation accuracy {:.4f}'.format(epoch,avg_loss_val,avg_acc_val))
        if avg_acc_val > best_val_acc:
            print("Saving the best model with valdation accuracy {}".format(avg_acc_val))
            best_val_acc = avg_acc_val
            best_state = model.state_dict()
            torch.save(model.state_dict(),best_model_path)
    torch.save(model.state_dict(),last_model_path)

    return best_val_acc, model, best_state
        


# In[24]:


#DCASE

def eval(config):
    
    if config.set.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    name_arr = np.array([])
    onset_arr = np.array([])
    offset_arr = np.array([])
    all_feat_files = [file for file in glob(os.path.join(config.path.test_w,'*.h5'))]

    for feat_file in all_feat_files:
        feat_name = feat_file.split('/')[-1]
        audio_name = feat_name.replace('h5','wav')

        print("Processing audio file : {}".format(audio_name))

        hdf_eval = h5py.File(feat_file,'r')
        strt_index_query =  hdf_eval['start_index_query'][:][0]
        onset,offset = evaluate_prototypes(config, hdf_eval, device, strt_index_query)

        name = np.repeat(audio_name,len(onset))
        name_arr = np.append(name_arr,name)
        onset_arr = np.append(onset_arr,onset)
        offset_arr = np.append(offset_arr,offset)

    df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
    csv_path = os.path.join(config.path.root,'Eval_out.csv')
    df_out.to_csv(csv_path,index=False)
    


# ## Test
# 

# In[25]:


initialize(job_name='test')


# In[26]:


cfg = compose(config_name='config')
s = Spectralizer(cfg)
f_ext = MyF_Ext(cfg, s)


# In[27]:


#f_ext.extract_features()


# In[28]:


#f_ext.extract_test()


# In[29]:


data_gen = Datagen(cfg)
X_train, Y_train, X_val, Y_val = data_gen.generate_train()
X_tr = torch.tensor(X_train)
Y_tr = torch.LongTensor(Y_train)
X_val = torch.tensor(X_val)
Y_val = torch.LongTensor(Y_val)
samples_per_cls = cfg.train.n_shot + cfg.train.n_query
batch_size_tr = samples_per_cls * cfg.train.k_way
batch_size_vd = batch_size_tr

num_batches_tr = len(Y_train)//batch_size_tr
num_batches_vd = len(Y_val)//batch_size_vd


samplr_train = RandomEpisodicSampler(Y_train,num_batches_tr,cfg.train.k_way, cfg.train.n_shot, cfg.train.n_query)
samplr_valid = RandomEpisodicSampler(Y_val,num_batches_vd,cfg.train.k_way,cfg.train.n_shot, cfg.train.n_query)

train_dataset = torch.utils.data.TensorDataset(X_tr,Y_tr)
valid_dataset = torch.utils.data.TensorDataset(X_val,Y_val)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_sampler=samplr_train,num_workers=0,pin_memory=True,shuffle=False)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_sampler=samplr_valid,num_workers=0,pin_memory=True,shuffle=False)


# In[30]:


model = Protonet()
best_acc,model,best_state = train(model,train_loader,valid_loader,cfg,num_batches_tr,num_batches_vd)
print("Best accuracy of the model on training set is {}".format(best_acc))


# In[32]:


eval(cfg)


# In[57]:


post_processing(cfg.path.data_test, os.path.join(cfg.path.root, 'Eval_out.csv'), 'pp.csv',
               cfg.train.n_shot)

