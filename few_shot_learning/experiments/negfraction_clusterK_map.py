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
    config.experiment.path.best_model = config.model
        
    scores = {}

    for fraction in tqdm(config.fractions):
        for cluster_K in tqdm(config.K):
            key = (fraction,cluster_K)
            scores[key] = {}
            for i in tqdm(range(config.iterations)):
                config.experiment.eval.fraction_neg = fraction
                if cluster_K == 1:
                    config.experiment.eval.clustering = False
                else:
                    config.experiment.eval.clustering = True
                    config.experiment.eval.cluster_K = cluster_K
                if config.dataset == 'TEST':
                    score = proteval.eval(None, None, config, None)
                else:
                    score = proteval.eval_help(None, None, config, None, 'VAL')
                scores[key][i] = score


    file = open('scores.pkl', 'wb')
    pickle.dump(scores, file)
    file.close()

@hydra.main(config_path='./', config_name='negfraction_clusterK_map_config')    
def main(config):
    run(config)
    
if __name__ == '__main__':
    main()


    
