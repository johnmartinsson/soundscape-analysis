import torch
import datasets.dcase_few_shot_bioacoustic as util
from torch.nn import functional as F

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
    dists = util.euclidean_dist(query_samples, prototypes)
    
    #Check
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    #.mean() -> 1/NcNq
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val


def ce_loss(pred, target):
    return torch.nn.functional.cross_entropy(pred, target)

def mixed_loss(pred, emb, target, n_support, config, supp_idxs=None):
    prot_loss = prototypical_loss(emb, target, n_support, supp_idxs)
    cross_loss = ce_loss(pred, target)
    loss = config.experiment.train.l*prot_loss + (1-config.experiment.train.l)*cross_loss
    return loss

def load(config):
    return mixed_loss