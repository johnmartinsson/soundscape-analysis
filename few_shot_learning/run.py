#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run an experiment."""

import sys
import pprint
#import logging
#logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
#                    level=logging.DEBUG,
#                    stream=sys.stdout)
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
        #module_evaluate      = utils.load_module(cfg['evaluate']['script_path'])

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(cfg)

        train_loader, valid_loader, test_loader = module_dataset.load_dataset(cfg['dataset'])

        model = module_model.load(cfg['model'])

        optimizer = module_optimizer.load(model.parameters(), cfg['optimizer'])
        loss_function = module_loss_function.load(cfg['loss_function'])


        #if 'initial_weights_path' in cfg['train']:
        #    model.load_weights(cfg['train']['initial_weights_path'])

        # training mode
        if mode == 'train':
            module_train.train(model, optimizer, loss_function, train_loader, valid_loader, cfg['train'])
        # evaluation mode
        if mode == 'evaluate':
            module_evaluate.evaluate(model, test_loader, cfg['evaluate'])

if __name__ == '__main__':
    args = utils.get_parser().parse_args()
    main(args.filename, args.mode)
