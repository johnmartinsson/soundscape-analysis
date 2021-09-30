import torch

def load(config, model):

    return torch.optim.Adam(model.parameters(), lr=config.train.lr)