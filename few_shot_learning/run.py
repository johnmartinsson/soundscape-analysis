#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run an experiment."""

import sys
import pprint
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)
import numpy as np

import utils

def main(yaml_filepath, mode):
    """Run experiments."""
    cfgs = utils.load_cfgs(yaml_filepath)
    print("Running {} experiments.".format(len(cfgs)))
    for cfg in cfgs:
        seed = int(cfg['train']['seed'])
        np.random.seed(seed)

        # Print the configuration - just to make sure that you loaded what you
        # wanted to load

        module_dataset       = utils.load_module(cfg['dataset']['script_path'])
        module_model         = utils.load_module(cfg['model']['script_path'])
        module_optimizer     = utils.load_module(cfg['optimizer']['script_path'])
        module_loss_function = utils.load_module(cfg['loss_function']['script_path'])
        module_train         = utils.load_module(cfg['train']['script_path'])
        module_eval          = utils.load_module(cfg['eval']['script_path'])

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(cfg)

        model = module_model.load(cfg['model'])
        #Should perhaps load different loaders depending on the mode
        train_loader, val_loader, test_loader = module_dataset.get_dataloaders(cfg['dataset'])
        optimizer = module_optimizer.load(cfg['optimizer'], model)
        loss_function = module_loss_function.load()
        train_function = module_train.load()
        eval_function = module_eval.load()

        #if 'initial_weights_path' in cfg['train']:
        #    model.load_weights(cfg['train']['initial_weights_path'])

        # training mode
        if mode == 'train':
            train_function(model, optimizer, loss_function, train_loader, val_loader, cfg)
        # evaluation mode
        if mode == 'evaluate':
            eval_function(model, test_loader, cfg)

def train(model, optimizer, loss_function, x_train, y_train, x_valid, y_valid, cfg):
    model.train()

    return model

def evaluate(model, x_test, y_test, cfg):
    model.eval()

    return


if __name__ == '__main__':
    args = utils.get_parser().parse_args()
    main(args.filename, args.mode)
