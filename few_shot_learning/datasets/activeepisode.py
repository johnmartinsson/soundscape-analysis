import torch.utils.data as data
import torch
import numpy as np
import datasets.dcase_few_shot_bioacoustic as util
import scipy
import logging
from collections import defaultdict

class ActiveQuerySampler(data.Sampler):
    
    #Include the option to choose the number of query samples
    #Y_train -> labels, just a list of the targets (list of ints?)
    def __init__(self, dataset, labels, n_episodes, n_way, n_support, n_query, device):
    
    
        #Number of episodes per epoch. len(labels)/(n_support * n_query) ?
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_samples = n_support+n_query
        self.dataset = dataset
        labels = np.array(labels)
        self.labels = labels
        self.sample_indices = []
        for i in range(max(labels) + 1):
            ix = np.argwhere(labels == i).reshape(-1)
            ix = torch.from_numpy(ix)
            self.sample_indices.append(ix)
            
        if self.n_way > len(self.sample_indices):
            #print(self.n_way)
            raise ValueError('Error: "n_way" parameter is higher than the unique number of classes')
            
        if device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def __len__(self):
        return self.n_episodes
    
    def __iter__(self):
        
        #first = True
        #self.params = list(self.model.parameters())
        
        #hehe
        self.max_entropy = 0
        self.min_entropy = 100
        
        #This is insanely slow right now :)
        
        for batch_nr in range(self.n_episodes):
            
            #Need to do stuff differently from here.
            #Select classes
            #Construct candidate set
            #Sample support + query.
            #Construct support sets as previously (five first examples per class respectively)
            #We need to construct prototypes here and do some loss shit on all other samples in the
            #candidate set
            #Construct a query set per class from the n_query hardest examples per class
            #This is slightly different from the paper where the sets are not balanced.
            #But let's do balanced for now. Might be easier.
            
            batch = []
            classes = torch.randperm(len(self.sample_indices))[:self.n_way]
            supp_idx = []
            q_idx = []
            for c in classes:
                #l is a list of indexes of elements in target belonging to class c
                l = self.sample_indices[c]
                perm = torch.randperm(len(l))
                supp_idx.append(l[perm[:self.n_support]])
                #Might consider making this just a teeny bit smaller for speed/VRAM
                #Perhaps configurable who knows
                q_idx.append(l[perm[self.n_support:300]])
                
            q_idx = np.array([e for l in q_idx for e in l])

            prototypes = []
            #Almost, need to do this per idx_l to stack proto separately
            for idx_l in supp_idx:
                tmp = torch.stack([self.dataset[i][0] for i in idx_l]).to(self.device)
                prototypes.append(self.model(tmp).mean(0))
            #t = torch.stack([self.dataset[i][0] for idx_l in supp_idx for i in idx_l])
            prototypes = torch.stack(prototypes)
            
            query = []
            i = 0
            while(i < len(q_idx)):
                if i + 50 > len(q_idx):
                    break
                else:
                    idx = np.arange(i,i+50)
                    tmp = torch.stack([self.dataset[j][0] for j in q_idx[idx]]).to(self.device)
                    query.append(self.model(tmp))
                    i += 50
            
            if i != len(q_idx):
                tmp = torch.stack([self.dataset[j][0] for j in q_idx[i:]]).to(self.device)
                query.append(self.model(tmp))
            query = torch.cat(query)
            
            prototypes = prototypes.detach().cpu()
            query = query.detach().cpu()
            
            #I'm not super pleased with this approach. Not very sensitive.
            #For example no real difference between a distance of 400 vs 600
            #TODO see this part over / track entropy change during training.
            dist = util.euclidean_dist(prototypes, query)
            #ones = torch.ones(dist.shape)
            #dist = torch.div(ones, dist)
            dist = dist.neg()
            p = torch.nn.functional.softmax(dist, dim=0).detach().numpy()
            entropy = [scipy.stats.entropy(p[:,i]) for i in range(p.shape[1])]
            
            if self.max_entropy < max(entropy):
                self.max_entropy = max(entropy)
            if self.min_entropy > min(entropy):
                self.min_entropy = min(entropy)

            entropy = list(zip(entropy, q_idx))
            entropy.sort(key=lambda l:l[0])
            hist_data = np.array([e[0] for e in entropy])
            #TODO save the entropy as a tensorboard histogram :)
            self.writer.add_histogram('Entropy: '+str(self.epoch), hist_data, batch_nr)
            
            d = defaultdict(list)
            for i in range(len(entropy)):
                label = self.labels[entropy[i][1]].item()
                if len(d[label]) == self.n_query:
                    continue
                else:
                    d[label] += [entropy[i][1]]
 
            batch = []
            for i in range(len(classes)):
                batch.append(torch.cat([supp_idx[i], torch.LongTensor(d[classes[i].item()])]))
                        
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
            
    def set_model(self, model):
        self.model = model
        
    def set_writer(self, writer):
        self.writer = writer
    
    def set_epoch(self, epoch):
        self.epoch = epoch