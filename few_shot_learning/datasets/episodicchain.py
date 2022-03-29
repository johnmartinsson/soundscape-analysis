

import numpy as np
import pandas as pd
import datasets.dcase_few_shot_bioacoustic as util
import torch

import pickle

'''
EpisodicChain logger object
'''
class EpisodicChain():
    
    def __init__(self, config, model, data_set):
        self.config = config
        self.model = model
        self.data_set = data_set
        self.chain = {}
        
        #These seems scrambled, is that really right?
        #I thought shuffle=false, perhaps just weird internal behaviour?
        self.x = self.data_set.tensors[0]
        self.y = self.data_set.tensors[1]
        
        if config.experiment.set.device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.ep = 0
        self.epoch = 0
    
    '''
    I want a prototype for all classes per episode. Does not to sample to heavily. Just ~20 samples per class should do (check what seems good enough in the end)?
    '''
    def log_episode(self, labels, emb):
           
        episode = {}
        
        #TODO: Make np
        episode['labels'] = labels
        episode['episode_emb'] = emb
        
        '''
        TODO: We might wanna do some processing on emb here instead of analysis notebook
        '''
        
        #Sampling prototype position and taking random query points in embedding space for all classes in base dataset.
        c_proto = {}
        q_point = {}
        
        classes = list(set(self.y.tolist()))
     
        
        for c in classes:
            ix = np.argwhere(c == np.array(self.y)).reshape(-1)
            if len(ix) > self.config.experiment.train.log_p_samples:
                ix_p = np.random.choice(ix, self.config.experiment.train.log_p_samples, replace=False)
                ix_n = list(set(ix).difference(ix_p))
                if len(ix_n) > self.config.experiment.train.log_q_samples:            
                    ix_q = np.random.choice(ix_n, self.config.experiment.train.log_q_samples, replace=False)
                else:
                    ix_q = ix_n
                    
            else:
                ix_q = None
                
            c_p = self.model(self.x[ix_p].to(torch.float).to(self.device))
            c_proto[c] = c_p.cpu().detach().mean(dim=0).numpy()
            
            if ix_q is not None:
                q = self.model(self.x[ix_q].to(torch.float).to(self.device))
                q_point[c] = q.cpu().detach().numpy()
        
        episode['prot_sample'] = c_proto
        episode['query_sample'] = q_point
        
        '''
        TODO: It is probably nice to include the prototypes and queries of the validation set.
              That could possibly give some indications of which way to sample episodes is best for down stream tasks
        '''
        
        self.chain[self.ep] = episode
        self.ep += 1
        
    def save_chain(self, scores):
        
        self.chain['scores'] = scores
        pickle.dump(self.chain, 'e'+str(self.epoch)+self.config.experiment.train.log_path)
        
        self.chain = {}
        self.epoch += 1