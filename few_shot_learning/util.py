import numpy as np
import torch

def time_2_frame(df,fps):


    #Margin of 25 ms around the onset and offsets
    #TODO: Should be in config

    df.loc[:,'Starttime'] = df['Starttime'] - 0.025
    df.loc[:,'Endtime'] = df['Endtime'] + 0.025

    #Converting time to frames

    start_time = [int(np.floor(start * fps)) for start in df['Starttime']]

    end_time = [int(np.floor(end * fps)) for end in df['Endtime']]

    return start_time,end_time

def class_to_int(labels):
    
    class_set = set(labels)
    ltoix = {label:index for index, label in enumerate(class_set)}
    return np.array([ltoix[label] for label in labels])

def balance_class_distribution(X,Y):

    '''  Class balancing through Random oversampling
    Args:
    -X: Feature
    -Y: labels

    Out:
    -X_new: Feature after oversampling
    -Y_new: Oversampled label list
    '''

    x_index = [[index] for index in range(len(X))]
    set_y = set(Y)


    ros = RandomOverSampler(random_state=42)
    x_unifm, y_unifm = ros.fit_resample(x_index, Y)
    unifm_index = [index_new[0] for index_new in x_unifm]

    X_new = np.array([X[index] for index in unifm_index])

    sampled_index = [idx[0] for idx in x_unifm]
    Y_new = np.array([Y[idx] for idx in sampled_index])

    return X_new,Y_new

def norm_params(X):

    '''  Normalize features
        Args:
        - X : Features

        Out:
        - mean : Mean of the feature set
        - std: Standard deviation of the feature set
        '''


    mean = np.mean(X)

    std = np.std(X)
    return mean, std

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        #we are currently getting stuck here?
        #why?
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)

    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)