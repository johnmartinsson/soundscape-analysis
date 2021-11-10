import torch

def load(parameters, cfg):
    optimizer = torch.optim.Adam(parameters, lr=float(cfg['learning_rate']))
    return optimizer
