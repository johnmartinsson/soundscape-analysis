
import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split
import datasets.dcase_few_shot_bioacoustic as util

#Relative import again

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
                self.y = util.class_to_int(self.labels)
            else:
                self.y = self.labels
            if config.datagen.balance:
                self.x, self.y = util.balance_class_distribution(self.x, self.y)
            
            array_train = np.arange(len(self.x))
            if config.datagen.stratify:
                _,_,_,_,train_array,valid_array = train_test_split(self.x, self.y, array_train,                                                     random_state=config.datagen.random_state, stratify=self.y)
            else:
                _,_,_,_,train_array,valid_array = train_test_split(self.x, self.y, array_train,                                                     random_state=config.datagen.random_state)
                
            self.train_index = train_array
            self.valid_index = valid_array
            if config.datagen.normalize:
                self.mean, self.std = util.norm_params(self.x[train_array])
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