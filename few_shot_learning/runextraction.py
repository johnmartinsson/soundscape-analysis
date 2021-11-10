import hydra
import datasets.feature_extract as fe
import os

'''
Game plan: Trawl code base for feature extraction thread.
           Remove this and place here instead.
           Mild config refactoring in main code base.
'''

#This should be it, plain and simple.
@hydra.main(config_path='config', config_name='extraction')
def main(config):
    
    #Create folder structure
    if not os.path.exists(config.path.output):
        os.mkdir(config.path.output)
    os.mkdir(os.path.join(config.path.output, 'hfiles'))
    os.mkdir(os.path.join(config.path.output, 'hfiles/train'))
    os.mkdir(os.path.join(config.path.output, 'hfiles/train/whole'))
    os.mkdir(os.path.join(config.path.output, 'hfiles/val'))
    os.mkdir(os.path.join(config.path.output, 'hfiles/test'))
    
    spectralizer = fe.Spectralizer(config)
    feature_extractor = fe.SpectralFeatureExtractor(config, spectralizer)
    feature_extractor.extract_features()

if __name__ == '__main__':
    main()