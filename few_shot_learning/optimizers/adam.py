import torch

def load(config, model):
    print('loading adam')
    return torch.optim.Adam(model.parameters(), lr=config.train.lr)