import torch
from torch.nn import functional as F
import math
import numpy as np

import datasets.dcase_few_shot_bioacoustic as util


def semi_prototypical_loss(input, target, n_support, config, semi_input=None, supp_idxs=None):
    
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    classes = torch.unique(target_cpu)
    n_labeled_classes = len(classes)
    n_all_classes = len(classes)
    n_query = target.eq(classes[0].item()).sum().item() - n_support
    if supp_idxs is None:
        #Rewrite, need to select only n_support. We might have n_query > n_support
        supp_idxs = list(map(lambda c: target_cpu.eq(c).nonzero()[:n_support].squeeze(1), classes))
        q_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    else:
        #Work from supp_idxs.
        q_idxs = None
    
    all_prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in supp_idxs])
    labeled_prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in supp_idxs])
    labeled_query_samples = input_cpu[q_idxs]
    
    all_query_samples = torch.zeros(0, input.shape[1])
    all_query_samples = torch.cat((all_query_samples, labeled_query_samples))
    
    
    if semi_input is not None:
        unlabeled_query_samples = torch.zeros(0, semi_input.shape[1])
        semi_input = semi_input.to('cpu')
        semi_prototypes = torch.zeros(0, semi_input.shape[1]) #correct right?
        i = 0
        n_semi_classes = 0
        while i < semi_input.shape[0]:
            semi_prototypes = torch.cat((semi_prototypes, 
                                         torch.reshape(semi_input[i:i+config.experiment.train.semi_n_shot].mean(0), (1, -1))), dim=0)
            n_semi_classes += 1
            unlabeled_query_samples = torch.cat((unlabeled_query_samples, semi_input[i+config.experiment.train.semi_n_shot:i+config.experiment.train.semi_n_shot+config.experiment.train.semi_n_query]))
            all_query_samples = torch.cat((all_query_samples, semi_input[i+config.experiment.train.semi_n_shot:i+config.experiment.train.semi_n_shot+config.experiment.train.semi_n_query]))
            
            i += config.experiment.train.semi_n_shot + config.experiment.train.semi_n_query
        
        all_prototypes = torch.cat((labeled_prototypes, semi_prototypes))
        n_all_classes += n_semi_classes
    else:
        unlabeled_query_samples = []
    

    
    dists = util.euclidean_dist(all_query_samples, all_prototypes)
    #All shapes look correct! 
    
    #For the loss, calculate a value for each example and add, then return mean.
    #Comp graph will do rest.
    loss_val = torch.zeros(0)
    
    '''
    Is the below implementation correct?
    Possible TODO: add some checks and balances for which distances we still enforce loss
    
    TODO: Implement a weighted average here as to not dilute the signal from the labeled data 
          when introducing more unlabeled learning samples.
    '''
    
    if config.experiment.train.semi_weighted:
        labeled_weight = 2
        unlabeled_weight = 0.5
    else:
        labeled_weight = 1
        unlabeled_weight = 1
    
    for i, x in enumerate(labeled_query_samples):
        prot_ix = math.floor(i/n_query)
        #print(F.softmax(-dists[i]))
        #print(torch.sum(F.softmax(-dists[i])))
        #print(F.softmax(-dists[i])[prot_ix])
        #print(F.softmax(-dists[i])[prot_ix].shape)
        loss_val = torch.cat((loss_val, labeled_weight*(-F.log_softmax(-dists[i])[prot_ix].reshape(1))))
        #denom = torch.sum(torch.tensor(map(torch.exp, -dists[i])))
    
    
    for i, x in enumerate(unlabeled_query_samples):
        prot_ix = math.floor(i/n_query)+len(labeled_prototypes)
        ix = list(range(0, len(labeled_prototypes))) + [prot_ix]
        loss_val = torch.cat((loss_val, unlabeled_weight*(-F.log_softmax(-dists[i+len(labeled_query_samples)][ix])[-1].reshape(1))))
    
    #TODO: Fix accuracy calc
    return loss_val.mean(), torch.tensor(0)
    
    '''
    Eh I don't think we will calculate the loss in this fashion anymore.
    
    
    labeled_log_p_y = F.log_softmax(-dists[0:n_query*n_labeled_classes], dim=1).view(n_all_classes, n_query, -1)
    print(labeled_log_p_y.shape)
    print(torch.sum(labeled_log_p_y[:,0,0]))
    
    labeled_target_inds = torch.arange(0, n_all_classes)
    labeled_target_inds = labeled_target_inds.view(n_all_classes, 1, 1)
    labeled_target_inds = labeled_target_inds.expand(n_all_classes, n_query, 1).long()
    #.mean() -> 1/NcNq
    labeled_loss_val = -labeled_log_p_y.gather(2, labeled_target_inds).squeeze().view(-1).mean()
    loss_val += labeled_loss_val
    _, labeled_y_hat = labeled_log_p_y.max(2)
    labeled_acc_val = labeled_y_hat.eq(labeled_target_inds.squeeze()).float().mean()
   
    return loss_val, acc_val
    '''
    return None

def load(config):
    return semi_prototypical_loss