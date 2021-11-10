'''
OBS: Deprecated. Does not work any more. Check code/config changes
'''


import sys
import os
sys.path.append('..')

import hydra
from hydra import compose, initialize
import pickle
from tqdm import tqdm

import eval.prototypical_eval as proteval
import datasets.dcase_few_shot_bioacoustic as util
import datasets.data_gen

def run(config):
    randomepisode_model_path = '/home/willbo/repos/soundscape-analysis/few_shot_learning/multirun/randomepisode2/14/best_model.pth'
    activequery_model_path = '/home/willbo/repos/soundscape-analysis/few_shot_learning/multirun/activequery/10/best_model.pth'

    if config.model == 'randomepisode':
        config.experiment.path.best_model = randomepisode_model_path
    elif config.model == 'activequery':
        config.experiment.path.best_model = activequery_model_path
    else:
        raise
        
    scores = {}

    for fraction in tqdm(config.fractions):
        scores[fraction] = {}
        for i in tqdm(range(config.iterations)):
            if config.vary_neg:
                config.experiment.eval.fraction_neg = fraction
                config.experiment.eval.fraction_pos = 1
            elif config.vary_pos:
                config.experiment.eval.fraction_neg = 1
                config.experiment.eval.fraction_pos = fraction
            score = proteval.eval(None, None, config, None)
            scores[fraction][i] = score


    file = open('scores.pkl', 'wb')
    pickle.dump(scores, file)
    file.close()

@hydra.main(config_path='./', config_name='negativeconfig')    
def main(config):
    run(config)
    
if __name__ == '__main__':
    main()


    
