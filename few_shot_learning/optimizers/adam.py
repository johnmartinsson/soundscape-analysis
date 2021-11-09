import torch

def load(config, model):
    print('loading adam')
    return torch.optim.Adam(model.parameters(), lr=config.experiment.train.lr, weight_decay=config.experiment.train.l2)