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


        
        


# In[9]:


'''
This could be an interface / abstract class to build audio 
to some other format instance to plug into feature extractor
'''



# In[10]:


'''
Possibly work on an raw files and annotations and return/write h5 files.
This might be clunky to include in a framework since this most likely is dataset dependent.
Might however benfit from having an interface which is inherited by classes working on specific datasets.
'''



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



# In[13]:


#Must somehow have access to all the data (just pass it).





#DCASE

#TODO Make it possible to use all samples as negatives.
#TODO Read up on this function, understand it better.



# ## Eval stuff
# 

# In[56]:


#DCASE pretty much

#Think about how to actually choose examples of the positive class in practice?
#Can one even do this? Read paper again




# ## Scoring

# In[ ]:


#Look at scoring files from dcase and try to understand them.


# ## Loop

# In[23]:


#DCASE



        


# In[24]:


#DCASE


    


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



# In[30]:


model = Protonet()
best_acc,model,best_state = train(model,train_loader,valid_loader,cfg,num_batches_tr,num_batches_vd)
print("Best accuracy of the model on training set is {}".format(best_acc))


# In[32]:


eval(cfg)


# In[57]:


post_processing(cfg.path.data_test, os.path.join(cfg.path.root, 'Eval_out.csv'), 'pp.csv',
               cfg.train.n_shot)

