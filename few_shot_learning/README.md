# Understanding the code
This is a brief overview of the code-base.

## Preparing the data
The data should be prepared by going to data/dcase_few_shot_bioaacoustics, some more info in the README.md, download the appropriate data files from the DCASE challenge:
- Development_Set.zip
- Evaluation_Set.zip
- Evaluation_Set_Full_Annotations.zip
and prepare them using the prepare_data.sh script.

## Extract time-frequency features from the data

    python runextraction.py
    
This will extract Mel spectrograms from the audio data, and save them in the hierarchical data format: HDF5. There will be one .h5 file for each waveform in the original data. (TODO: explain how to access the data in these files, annotations, features, etc).

## Running a training script
    python runhydra.py hydra.run.dir=outputs/expout root_path=/home/john/gits/soundscape-analysis/few_shot_learning/ +experiment=randomepisode experiment.train.epochs=30 experiment.set.train=true experiment.path.features=/home/john/gits/soundscape-analysis/few_shot_learning/data/dcase_few_shot_bioacoustic experiment.path.data_val=/home/john/gits/soundscape-analysis/few_shot_learning/data/dcase_few_shot_bioacoustic/val experiment.path.data_test=/home/john/gits/soundscape-analysis/few_shot_learning/data/dcase_few_shot_bioacoustic/test experiment.path.val_OG=/home/john/gits/soundscape-analysis/few_shot_learning/data/dcase_few_shot_bioacoustic/Validation_Set experiment.path.test_OG=/home/john/gits/soundscape-analysis/few_shot_learning/data/dcase_few_shot_bioacoustic/Test_Set
    
The __hydra.run.dir__ sets the path for the output of the training procedure, this is where the trained model and other logging during training will be stored, this is also the path which should be set if you want to evaluate a model when setting __experiment.set.train=false__. The __root_path__ is simply the absolute path to the few_shot_learning code (where this README.md resides). The rest of the paths point to the train/val/test features. These config options can also be set in the __config/experiment/<experiment_name>.yaml__ file if you prefer to not write this out each time. The experiment configuration to use is specified by __+experiment <experiment_name>__ where __<experiment_name>__ is __randomepisode__ in this case.

## Evaluating a trained model
Assuming that all paths have been set in the appropriate config file, and that your working directory is the one where this README.md file resides.

    python runhydra.py hydra.run.dir=outputs/expout root_path=$(pwd) +experiment=randomepisode experiment.train.epochs=30 experiemnt.set.train=false
    
This will evaluate the model on the test data.


# Few-shot learning

A collection of few-shot learning code.

In a __K__-way __N__-shot problem we have access to a support set with __N__ examples in some space __X__ of each of the __K__ classes, and we want to learn how to classify unseen query points __q__ from the same space __X__ from these few examples. 

A typical approach for few-shot learning is to learn a feature extractor __r = f(x)__ where __r__ is the representation of the input __x__, and assign the class of the query point __q__ according to closeness to the support of each class in this representation space. E.g., nearest neighbour for the mean representation of each class support.

## Useful libraries and github repos
- [few-shot](https://github.com/oscarknagg/few-shot)
  - implementations of various few-shot learning methods
- [few-shot bioacoustic baseline](https://github.com/c4dm/dcase-few-shot-bioacoustic/tree/main/baselines/deep_learning)
  - baseline implementation of prototypical networks used in [DCASE_2021_task5](http://dcase.community/challenge2021/task-few-shot-bioacoustic-event-detection)

## Ideas
- Capture the variance in the representation space. Early methods in few-shot learning, such as prototypical networks, represent the support of a class as the average of the class support in the representation space. It seems reasonable to assume that the variance of the representations for each class support is different, i.e., that they are more or less spread out around their average. Papers which seem to have explored this idea are:
  - [Infinite Mixture Prototypes for Few-shot Learning](http://proceedings.mlr.press/v97/allen19b.html)
  - [Variational Few-Shot Learning](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Variational_Few-Shot_Learning_ICCV_2019_paper.html)
  - I did not know that people had done this already, and I like these ideas. Is there anything else to do in this direction? Or is it exhausted.

## Example of how to run a script
python runhydra.py hydra.run.dir=outputs/expout root_path=/home/willbo/repos/soundscape-analysis/few-shot/ +experiment=randomepisode experiment.train.epochs=30
