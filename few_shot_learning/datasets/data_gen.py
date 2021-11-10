
import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split
import datasets.dcase_few_shot_bioacoustic as util

#Relative import again


'''
Should now return just X_train and Y_train. None of this training data will be used in a validation loader.
This file should probably sooner or later have a different name for integration into the repo.
'''
class Datagen():
    
    def __init__(self, config):
        
        self.config = config
        
        if config.experiment.datagen.raw:
            #These obviosly requires more processing down the pipe but that is application dependent.
            #Leave be for now
            hf = h5py.File(os.path.join(config.experiment.path.train_features, 'raw_train.h5'))
        else:
            hf = h5py.File(os.path.join(config.experiment.path.train_features, 'mel_train.h5'))
            self.x = hf['features'][:]
            self.labels = [s.decode() for s in hf['labels'][:]]
            if config.experiment.datagen.ltoi:
                self.y = util.class_to_int(self.labels)
            else:
                self.y = self.labels
            if config.experiment.datagen.balance:
                self.x, self.y = util.balance_class_distribution(self.x, self.y)
             
            if config.experiment.datagen.normalize:
                self.mean, self.std = util.norm_params(self.x)
            else:
                self.mean = None
                self.std = None
                
    def feature_scale(self, x):
        return (x - self.mean)/self.std
    
    def generate_train(self):
        
        X_train = self.x
        Y_train = self.y
        if self.config.experiment.datagen.normalize:
            X_train = self.feature_scale(X_train)
            
        return X_train, Y_train
        

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
        if self.config.experiment.datagen.normalize:
            X_pos = self.feature_scale(X_pos)
            X_neg = self.feature_scale(X_neg)
            X_query = self.feature_scale(X_query)
            
        return X_pos, X_neg, X_query