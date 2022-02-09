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
from datasets import dicts
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from sklearn.model_selection import KFold
import utils

#Perhaps we should consider moving over to hydra?
@hydra.main(config_path='config', config_name='config')
def main(cfg):
    """Run experiments."""
      
    # Print the configuration - just to make sure that you loaded what you
    # wanted to load

    module_dataset       = utils.load_module(cfg.experiment.dataset.script_path)
    module_model         = utils.load_module(cfg.experiment.model.script_path)
    module_optimizer     = utils.load_module(cfg.experiment.optimizer.script_path)
    module_loss_function = utils.load_module(cfg.experiment.loss_function.script_path)
    module_train         = utils.load_module(cfg.experiment.train.script_path)
    module_eval          = utils.load_module(cfg.experiment.eval.script_path)

    #Make this look pretty somehow
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
    
    folder = None
    gen_folds = None
    class_map = None
    class_dict = None
    tr_cls_keys = None
    val_cls_keys = None
    fold = None
    
    if cfg.experiment.set.trainvalcv:
        
        class_map, class_dict = dicts.get_dicts(cfg)
        folder = KFold(n_splits=cfg.experiment.train.folds, shuffle=True, random_state=cfg.seed)
        gen_folds = folder.split(list(class_map.keys()))
        
    
    optimizer = module_optimizer.load(cfg, model)
    loss_function = module_loss_function.load(cfg)
    train_function = module_train.load()
    
    if cfg.experiment.set.trainvalcv:
        eval_function = module_eval.load_TrainValCV()
    else:
        eval_function = module_eval.load()

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
    
    if cfg.experiment.set.trainvalcv:
        
        #We might wanna save some data here, what classes are in each fold etc....
        
        fold = 0
        #TODO: More stuff to do with the folds here. We wanna do this shit once per fold with different writer names.
        for tr_cls_ix, val_cls_ix in gen_folds:
            writer = SummaryWriter(comment=cfg.writer.comment+'fold_'+str(fold))
            if cfg.experiment.set.train:
                tr_cls_keys = np.array(list(class_map.keys()))[tr_cls_ix]
                val_cls_keys = np.array(list(class_map.keys()))[val_cls_ix]
                
                train_loader, val_loader = module_dataset.get_dataloaders_TrainValCV(cfg, class_map, class_dict, tr_cls_keys)
                train_function(model, optimizer, loss_function, train_loader, val_loader, cfg, writer, fold, class_map, class_dict, tr_cls_keys, val_cls_keys)
                
            if cfg.experiment.set.eval:
                #Not entirely nessecary to pass all these things when working on the test set but whatever.
                eval_function(model, test_loader, cfg, writer, fold, class_map, class_dict, tr_cls_keys, val_cls_keys)
                
            fold += 1
    else:
        if cfg.experiment.set.train:
            train_function(model, optimizer, loss_function, train_loader, val_loader, cfg, writer)
        if cfg.experiment.set.eval:
            eval_function(model, test_loader, cfg, writer)
    # evaluation mode
    
    
    if writer is not None:
        writer.close()
    
if __name__ == '__main__':
    main()
