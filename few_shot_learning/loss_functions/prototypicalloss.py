import torch
from torch.nn import functional as F

import datasets.dcase_few_shot_bioacoustic as util

'''
This function might need some rewriting.
We dont really have targets for the unsupervised sequence of segments.
Can't really think of a way to do this either. Perhaps we can create that here actually.
If we know the structure of the unsupervised data and has that as input then we can create it here.
Should work?

It looks to me that the loss right now is build around everything being in order.
If we keep this we should be good to go.
'''

def prototypical_loss(input, target, n_support, config, semi_input=None, supp_idxs=None):
    
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
    
    '''
    I think the unlabeled data can fit here.
    Create some prototypes and concatenate those to prototypes.
    Somehow add stuff to the classes list
    
    I think here that the tensor is just 2D is fine.
    Have all of the in row. index 0 upto and including 9 for examples is one event with examples. 0-4 proto, 5-9 query.
    Check config options for how to divide this array into prototypes and queries. Count the number of additional classes etc.
    '''
    
    #This is a huge training bottleneck :), ideas to speed this part up.
    #Bottleneck could be the code below this block as well.
    if semi_input is not None:
        semi_input = semi_input.to('cpu')
        semi_prototypes = torch.zeros(0, semi_input.shape[1]) #correct right?
        i = 0
        semi_classes = 0
        while i < semi_input.shape[0]:
            semi_prototypes = torch.cat((semi_prototypes, 
                                         torch.reshape(semi_input[i:i+config.experiment.train.semi_n_shot].mean(0), (1, -1))), dim=0)
            semi_classes += 1
            query_samples = torch.cat((query_samples, semi_input[i+config.experiment.train.semi_n_shot:i+config.experiment.train.semi_n_shot+config.experiment.train.semi_n_query]))
            
            i += config.experiment.train.semi_n_shot + config.experiment.train.semi_n_query
        prototypes = torch.cat((prototypes, semi_prototypes))
        n_classes += semi_classes
        
    dists = util.euclidean_dist(query_samples, prototypes)
    
    #Where does the labels fit, how do we tack on more pseudo labels for the unlabeled data?
    #Take note here that the same number of queries need to be taken for the labeled data as for the semi supervised 
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    #.mean() -> 1/NcNq
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val
    
def load(config):
    return prototypical_loss