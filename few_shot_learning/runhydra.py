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

import utils

#Perhaps we should consider moving over to hydra?
@hydra.main(config_name='config')
def main(cfg):
    """Run experiments."""
    #cfgs = utils.load_cfgs(yaml_filepath)
    #print("Running {} experiments.".format(len(cfgs)))
    #for cfg in cfgs:

    #cfg = utils.make_paths_absolute(os.getcwd(), cfg)
        
    seed = int(cfg['train']['seed'])
    np.random.seed(seed)

    # Print the configuration - just to make sure that you loaded what you
    # wanted to load

    module_dataset       = utils.load_module(cfg.dataset.script_path)
    module_model         = utils.load_module(cfg['model']['script_path'])
    module_optimizer     = utils.load_module(cfg['optimizer']['script_path'])
    module_loss_function = utils.load_module(cfg['loss_function']['script_path'])
    module_train         = utils.load_module(cfg['train']['script_path'])
    module_eval          = utils.load_module(cfg['eval']['script_path'])

    #Make this look pretty somehow
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(cfg)
    #print(cfg)
    #pyaml.dump(cfg)
    print(OmegaConf.to_yaml(cfg))

    model = module_model.load(cfg['model'])
    #Should perhaps load different loaders depending on the mode
    if cfg.set.train:
        train_loader, val_loader = module_dataset.get_dataloaders_train(cfg)
        print(train_loader)
        print(val_loader)
    if cfg.set.eval:
        test_loader = module_dataset.get_dataloaders_test(cfg)
    #train_loader, val_loader, test_loader = module_dataset.get_dataloaders(cfg['dataset'])
    optimizer = module_optimizer.load(cfg, model)
    loss_function = module_loss_function.load()
    train_function = module_train.load()
    eval_function = module_eval.load()

    #if 'initial_weights_path' in cfg['train']:
    #    model.load_weights(cfg['train']['initial_weights_path'])

    # training mode
    if cfg.set.train:
        train_function(model, optimizer, loss_function, train_loader, val_loader, cfg)
    # evaluation mode
    if cfg.set.eval:
        eval_function(model, test_loader, cfg)


if __name__ == '__main__':
    main()
