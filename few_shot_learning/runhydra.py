#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run an experiment."""

import sys
import pprint
import hydra
from omegaconf import OmegaConf
import pyaml
import os
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

import utils

#Perhaps we should consider moving over to hydra?
@hydra.main(config_path='config', config_name='config')
def main(cfg):
    """Run experiments."""
    #cfgs = utils.load_cfgs(yaml_filepath)
    #print("Running {} experiments.".format(len(cfgs)))
    #for cfg in cfgs:

    #cfg = utils.make_paths_absolute(os.getcwd(), cfg)
      
    # Print the configuration - just to make sure that you loaded what you
    # wanted to load

    module_dataset       = utils.load_module(cfg.experiment.dataset.script_path)
    module_model         = utils.load_module(cfg.experiment.model.script_path)
    module_optimizer     = utils.load_module(cfg.experiment.optimizer.script_path)
    module_loss_function = utils.load_module(cfg.experiment.loss_function.script_path)
    module_train         = utils.load_module(cfg.experiment.train.script_path)
    module_eval          = utils.load_module(cfg.experiment.eval.script_path)

    #Make this look pretty somehow
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(cfg)
    #print(cfg)
    #pyaml.dump(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    #Perhaps not best way to seed but.
    if cfg.seed != -1:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    
    model = module_model.load(cfg)
    #Should perhaps load different loaders depending on the mode
    if cfg.experiment.set.train:
        train_loader, val_loader = module_dataset.get_dataloaders_train(cfg)
    if cfg.experiment.set.eval:
        test_loader = module_dataset.get_dataloaders_test(cfg)
    #train_loader, val_loader, test_loader = module_dataset.get_dataloaders(cfg['dataset'])
    optimizer = module_optimizer.load(cfg, model)
    loss_function = module_loss_function.load()
    train_function = module_train.load()
    eval_function = module_eval.load()

    #if 'initial_weights_path' in cfg['train']:
    #    model.load_weights(cfg['train']['initial_weights_path'])
    
    #This amount of configurability is probably enough for now.
    if cfg.writer.directory != 'none':
        writer = SummaryWriter(log_dir=cfg.writer.directory)
    elif cfg.writer.comment != 'none':
        writer = SummaryWriter(comment=cfg.writer.comment)
    else:
        writer = SummaryWriter()
    
    '''
    TODO: Add config options for hydra settings such as output directory
    '''
    
    # training mode
    if cfg.experiment.set.train:
        train_function(model, optimizer, loss_function, train_loader, val_loader, cfg, writer)
    # evaluation mode
    if cfg.experiment.set.eval:
        eval_function(model, test_loader, cfg, writer)


if __name__ == '__main__':
    main()