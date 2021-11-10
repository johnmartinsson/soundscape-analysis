import numpy as np
import scipy
import mir_eval
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import datasets.feature_extract as fe
import datasets.randomepisode as re
import datasets.activeepisode as ae
import datasets.data_gen as dg
import datasets.smoothquery as sq
import models.prototypical as pt
from sklearn.cluster import  KMeans

import pandas as pd
import os
import utils
import math
import json
import csv
from datetime import datetime
import copy
from scipy import stats
from glob import glob


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

#Possible that this is something that could be of more general use.
def get_probability(pos_proto,neg_proto,query_set_out):


    """Calculates the  probability of each query point belonging to either the positive or negative class
     Args:
     - x_pos : Model output for the positive class
     - neg_proto : Negative class prototype calculated from randomly chosed 100 segments across the audio file
     - query_set_out:  Model output for the first 8 samples of the query set

     Out:
     - Probabiility array for the positive class
     """

    #pos_prototype = x_pos.mean(0)
    prototypes = torch.stack([pos_proto,neg_proto])
    dists = euclidean_dist(query_set_out,prototypes)
    #Eh what. Is done in the OG code as well?
    '''  Taking inverse distance for converting distance to probabilities'''
    inverse_dist = torch.div(1.0, dists)
    prob = torch.softmax(inverse_dist,dim=1)
    '''  Probability array for positive class'''
    prob_pos = prob[:,0]

    return prob_pos.detach().cpu().tolist()

def get_probability_negdistance(pos_proto,neg_proto,query_set_out):


    """Calculates the  probability of each query point belonging to either the positive or negative class
     Args:
     - x_pos : Model output for the positive class
     - neg_proto : Negative class prototype calculated from randomly chosed 100 segments across the audio file
     - query_set_out:  Model output for the first 8 samples of the query set

     Out:
     - Probabiility array for the positive class
     """
    
    '''
    This function should now to what we want right?
    Should we always use this? I'm inclined to say yes.
    Think this is better than actually using 1/d and softmax on that.
    '''
    
    #pos_prototype = x_pos.mean(0)
    #TODO: Configurable emb_dim
    prototypes = torch.zeros(0, pos_proto.shape[0])
    prototypes = prototypes.to('cuda') #TODO: Fix this
    prototypes = torch.cat((prototypes, torch.reshape(pos_proto, (1, -1))))
    if len(neg_proto.shape) > 1:
        prototypes = torch.cat((prototypes, neg_proto))
    else:
        prototypes = torch.cat((prototypes, torch.reshape(neg_proto, (1, -1))))
    dists = euclidean_dist(query_set_out,prototypes)
    dists = torch.neg(dists)
    #Eh what. Is done in the OG code as well?
    '''  Taking inverse distance for converting distance to probabilities'''
    prob = torch.softmax(dists,dim=1)
    '''  Probability array for positive class'''
    prob_pos = prob[:,0]

    return prob_pos.detach().cpu().tolist()



#Quite possible that this should actually be somewhere else.

def evaluate_prototypes(config=None,hdf_eval=None,device= None,strt_index_query=None, model=None):

    """ Run the evaluation
    Args:
     - conf: config object
     - hdf_eval: Features from the audio file
     - device:  cuda/cpu
     - str_index_query : start frame of the query set w.r.t to the original file

     Out:
     - onset: Onset array predicted by the model
     - offset: Offset array predicted by the model
      """
    hop_seg = int(config.experiment.features.hop_seg * config.experiment.features.sr // config.experiment.features.hop_mel)

    gen_eval = dg.TestDatagen(hdf_eval,config)
    X_pos, X_neg,X_query = gen_eval.generate_eval()

    X_pos = torch.tensor(X_pos)
    Y_pos = torch.LongTensor(np.zeros(X_pos.shape[0]))
    X_neg = torch.tensor(X_neg)
    Y_neg = torch.LongTensor(np.zeros(X_neg.shape[0]))
    X_query = torch.tensor(X_query)
    Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

    num_batch_query = len(Y_query) // config.experiment.eval.query_batch_size
    
    if config.experiment.eval.smoothquery:
        #Rough edges for now
        query_dataset = sq.SmoothQuerySet(X_query, config.experiment.eval.smoothing)
        b_size = math.floor(config.experiment.eval.query_batch_size/5)
        q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,batch_size=b_size,shuffle=False)
    else:
        query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
        q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,batch_size=config.experiment.eval.query_batch_size,shuffle=False)
    
    if model is None:
        #This should also listen to the config for which model we uses.
        module_model = utils.load_module(config.experiment.model.script_path)
        model = module_model.load(config)

        if device == 'cpu':
            model.load_state_dict(torch.load(config.experiment.path.best_model, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(config.experiment.path.best_model))

    model.to(device)
    model.eval()

    'List for storing the combined probability across all iterations'
    prob_comb = []
    
    #TODO: Remove this iteration stuff probably? What's the point?
    #Perhaps as a precaution do an empirical test on doing this versus just increasing the sample size to * iterations.
    iterations = config.experiment.eval.iterations
    num_neg_prot = 0
    for i in range(iterations):
        prob_pos_iter = []
        
        '''
        Selecting negative examples from which we construct the negative prototype
        '''
        if config.experiment.eval.samples_neg != -1 or (config.experiment.eval.use_fraction_neg and config.experiment.eval.fraction_neg != 1):
            if config.experiment.eval.use_fraction_neg:
                neg_indices = torch.randperm(len(X_neg))[:math.floor(config.experiment.eval.fraction_neg*len(X_neg))]
            else:
                neg_indices = torch.randperm(len(X_neg))[:config.experiment.eval.samples_neg]
        else:
            neg_indices = torch.tensor(range(len(X_neg)))
        X_neg = X_neg[neg_indices]
        Y_neg = Y_neg[neg_indices]
        batch_size_neg = config.experiment.eval.negative_set_batch_size
        neg_dataset = torch.utils.data.TensorDataset(X_neg, Y_neg)
        negative_loader = torch.utils.data.DataLoader(dataset=neg_dataset, batch_sampler=None, batch_size=batch_size_neg)
        
        '''
        Selecting positive examples from which we construct the positive prototype.
        '''
        if config.experiment.eval.samples_pos != -1 or (config.experiment.eval.use_fraction_pos and config.experiment.eval.fraction_pos != 1):
            if config.experiment.eval.use_fraction_pos:
                pos_indices = torch.randperm(len(X_pos))[:math.floor(config.experiment.eval.fraction_pos*len(X_pos))]
            else:
                pos_indices = torch.randperm(len(X_pos))[:config.experiment.eval.samples_pos]
        else:
            pos_indices = torch.tensor(range(len(X_pos)))
        X_pos = X_pos[pos_indices]
        Y_pos = Y_pos[pos_indices]
        batch_size_pos = config.experiment.eval.positive_set_batch_size
        pos_dataset = torch.utils.data.TensorDataset(X_pos, Y_pos)
        pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_sampler=None, batch_size=batch_size_pos)

        neg_iterator = iter(negative_loader)
        pos_iterator = iter(pos_loader)
        q_iterator = iter(q_loader)

        if config.experiment.eval.samples_neg != -1  or (config.experiment.eval.use_fraction_neg and config.experiment.eval.fraction_neg != 1):
            print("Iteration number {}".format(i))
        
        #TODO: Implement clustering and provide several negative prototypes if wanted.
        #Make this highly configurable. We might even want to split this into files/function calls and have some if structure here instead?
        #So if no clustering we return shape (emb_dim)
        #Else return shape (K, emb_dim)
        #The get probability function can work with both
        
        neg_set_feat = torch.zeros(0,1024).cpu()
        print('Processing negatives')
        for batch in tqdm(neg_iterator):
            x_neg, y_neg = batch
            x_neg = x_neg.to(device)
            #True baind-aid fix right here:
            if config.type.classifier:
                feat_neg, _ = model(x_neg)
            else:
                feat_neg = model(x_neg)
            feat_neg = feat_neg.detach().cpu()
            neg_set_feat = torch.cat((neg_set_feat, feat_neg), dim=0)
        
        #Does this work? Eh I think it does actually, first try ezi pizi #NOPE
        if config.experiment.eval.clustering:
            if config.experiment.eval.cluster_method == 'kmeans':
            
                cluster = KMeans(config.experiment.eval.cluster_K)
                print('Fitting cluster')
                cluster.fit(neg_set_feat)
                neg_proto = torch.tensor(cluster.cluster_centers_)
                neg_proto = neg_proto.to(device)
                num_neg_prot = len(cluster.cluster_centers_)
        else:
            neg_proto = neg_set_feat.mean(dim=0)
            neg_proto =neg_proto.to(device)
            num_neg_prot = 1
        
        
        pos_set_feat = torch.zeros(0,1024).cpu()
        print('Processing positives')
        for batch in tqdm(pos_iterator):
            x_pos, y_pos = batch
            x_pos = x_pos.to(device)
            if config.type.classifier:
                feat_pos, _ = model(x_pos)
            else:
                feat_pos = model(x_pos)
            feat_pos = feat_pos.detach().cpu()
            pos_set_feat = torch.cat((pos_set_feat, feat_pos), dim=0)
        pos_proto = pos_set_feat.mean(dim=0)
        pos_proto =pos_proto.to(device)
        
        print('Processing queries')
        #Just tape 'smooth query' into here for now, quick testing.
        #A batch is 4d here. (batch_size, smoothing, -1, -1) -> (batch_size, -1)
        if config.experiment.eval.smoothquery:
            for batch in tqdm(q_iterator):
                x_q = batch
                x_q = x_q.to(device)
                #outer dimension
                embedded = torch.zeros(0,1024)
                embedded = embedded.to(device)
                for i in range(len(x_q)):
                    tmp = model(x_q[i])
                    embedded = torch.cat((embedded, tmp.mean(dim=0).reshape(1,-1)), dim=0)
                probability_pos = get_probability_negdistance(pos_proto, neg_proto, embedded)
                prob_pos_iter.extend(probability_pos)    
        else:
            
            for batch in tqdm(q_iterator):
                x_q, y_q = batch
                x_q = x_q.to(device)
                if config.type.classifier:
                    x_query, _ = model(x_q)
                else:
                    x_query = model(x_q)
                #TODO: Expand this for several negative prototypes.
                #probability_pos = get_probability(pos_proto, neg_proto, x_query)
                probability_pos = get_probability_negdistance(pos_proto, neg_proto, x_query)
                prob_pos_iter.extend(probability_pos)
            
        

        prob_comb.append(prob_pos_iter)
        
        if config.experiment.eval.samples_neg == -1 or (config.experiment.eval.use_fraction_neg and config.experiment.eval.fraction_neg == 1): 
            break

    prob_final = np.mean(np.array(prob_comb),axis=0)

    krn = np.array([1, -1])
    #If using clustering we might want to consider scaling the threshold
    #Lets test this for now.
    #prob_thresh = np.where(prob_final > config.experiment.eval.p_thresh, 1, 0)
    prob_thresh = np.where(prob_final > 1/(1+num_neg_prot), 1, 0)
    
    prob_pos_final = prob_final * prob_thresh
    changes = np.convolve(krn, prob_thresh)

    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_time_query = strt_index_query * config.experiment.features.hop_mel / config.experiment.features.sr

    onset = (onset_frames + 1) * (hop_seg) * config.experiment.features.hop_mel / config.experiment.features.sr
    onset = onset + str_time_query

    offset = (offset_frames + 1) * (hop_seg) * config.experiment.features.hop_mel / config.experiment.features.sr
    offset = offset + str_time_query

    assert len(onset) == len(offset)
    return onset, offset


'''
One can argue that this code should not be here maybe?
This is quite meta-learning heavy.
We could also just make a copy of this file. This is very much not neet.
Argueably better to start splitting it. Remove all that has to do with meta-learning
such as this function and keep everything that is relevant for SED.
'''
def get_dataloaders_train(config):

    print('get_dataloaders')

    datagen = dg.Datagen(config)
    #X_train, Y_train, X_val, Y_val = datagen.generate_train()
    X_train, Y_train = datagen.generate_train()

    X_tr = torch.tensor(X_train)
    Y_tr = torch.LongTensor(Y_train)
    #X_val = torch.tensor(X_val)
    #Y_val = torch.LongTensor(Y_val)

    samples_per_cls = config.experiment.train.n_shot + config.experiment.train.n_query
    batch_size_tr = samples_per_cls * config.experiment.train.k_way
    #batch_size_vd = batch_size_tr
    num_batches_tr = len(Y_train)//batch_size_tr
    #num_batches_vd = len(Y_val)//batch_size_vd
    
    
    train_set = torch.utils.data.TensorDataset(X_tr, Y_tr)
    #val_set = torch.utils.data.TensorDataset(X_val, Y_val)
    
    if config.experiment.train.sampler == 'random':
        tr_sampler = re.RandomEpisodicSampler(Y_train, num_batches_tr, config.experiment.train.k_way,
        config.experiment.train.n_shot, config.experiment.train.n_query)
        #val_sampler = re.RandomEpisodicSampler(Y_val, num_batches_vd, config.experiment.train.k_way,
        #config.experiment.train.n_shot, config.experiment.train.n_query)

    if config.experiment.train.sampler == 'activequery':
        tr_sampler = ae.ActiveQuerySampler(train_set, Y_train, num_batches_tr, config.experiment.train.k_way,
        config.experiment.train.n_shot, config.experiment.train.n_query, config.experiment.set.device, config.experiment.train.query_candidates,
                                          config.type.classifier)
        #val_sampler = re.RandomEpisodicSampler(Y_val, num_batches_tr, config.experiment.train.k_way,
        #config.experiment.train.n_shot, config.experiment.train.n_query)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_sampler=tr_sampler, num_workers=0,
    pin_memory=True, shuffle=False)
    #val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_sampler=val_sampler, num_workers=0,
    #pin_memory=True, shuffle=False)

    return train_loader, None

#For the dcase acoustic this can just probably return None
#Handles data loading differently for eval
#Check prototypical_eval.py
def get_dataloaders_test(config):    
    return None

    
    
########################################################################################
########################################################################################
########################## EVAL/METRIC PIPE HERE, SHITLOAD OF CODE #####################
########################################################################################
########################################################################################

#For selection of positives to be used?
def dummy_choice(csv, n_shots):
    events = []
    for i in range(len(csv)):
                if(csv.loc[i].values[-1] == 'POS' and len(events) < n_shots):
                    events.append(csv.loc[i].values)
    return events

#Consider own file for this!
#Should we use validation or test data for this?????
#Test data most likely?
#Since eval is done on test data right?
def post_processing(test_path, evaluation_file, new_evaluation_file, n_shots=5):
    '''Post processing of a prediction file by removing all events that have shorter duration
    than 60% of the minimum duration of the shots for that audio file.
    
    Parameters
    ----------
    val_path: path to validation set folder containing subfolders with wav audio files and csv annotations
    evaluation_file: .csv file of predictions to be processed
    new_evaluation_file: .csv file to be saved with predictions after post processing
    n_shots: number of available shots
    '''
    
    '''
    I think it is of great interest to not just choose the first five positives in practice.
    Sure this is part of the challenge. But... Interesting to invesigate. Discussion about growing
    number of supports can fit here to?
    
    Perhaps we can add some post-processing smoothing. That is if there is a small gap in what is believed to be an event than this is converted.
    ________****_*****___________ -> _______*********_________
    '''
    
    print(test_path)
    csv_files = [file for file in glob(os.path.join(test_path, '*.csv'))]
    
    dict_duration = {}
    for csv_file in csv_files:
        audiofile = csv_file.replace('.csv', '.wav')
        df = pd.read_csv(csv_file)
        events = dummy_choice(df, n_shots)
        min_duration = 10000 #configurable?
        for event in events:
            if float(event[2])-float(event[1]) < min_duration:
                min_duration = float(event[2])-float(event[1])
        #dict_duration[audiofile] = min_duration
        dict_duration[os.path.split(audiofile)[1]] = min_duration

    results = []
    with open(evaluation_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            results.append(row)

    new_results = [['Audiofilename', 'Starttime', 'Endtime']]
    for event in results:
        audiofile = os.path.split(event[0])[1]
        min_dur = dict_duration[audiofile]
        if float(event[2])-float(event[1]) >= 0.6*min_dur:
            new_results.append([os.path.split(event[0])[1], event[1], event[2]])

    with open(new_evaluation_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_results)


def fast_intersect(ref, est):
    """Find all intersections between reference events and estimated events (fast).
    Best-case complexity: O(N log N + M log M) where N=length(ref) and M=length(est)

    Parameters
    ----------
    ref: np.ndarray [shape=(2, n)], real-valued
         Array of reference events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.

    est: np.ndarray [shape=(2, m)], real-valued
         Array of estimated events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.


    Returns
    -------
    matches: list of sets, length n, integer-valued
         Property: matches[i] contains the set of all indices j such that
            (ref[0, i]<=est[1, j]) AND (ref[1, i]>=est[0, j])
    """
    ref_on_argsort = np.argsort(ref[0, :])
    ref_off_argsort = np.argsort(ref[1, :])

    est_on_argsort = np.argsort(est[0, :])
    est_off_argsort = np.argsort(est[1, :])

    est_on_maxindex = est.shape[1]
    est_off_minindex = 0
    estref_matches = [set()] * ref.shape[1]
    refest_matches = [set()] * ref.shape[1]
    for ref_id in range(ref.shape[1]):
        ref_onset = ref[0, ref_on_argsort[ref_id]]
        est_off_sorted = est[1, est_off_argsort[est_off_minindex:]]
        search_result = np.searchsorted(est_off_sorted, ref_onset, side="left")
        est_off_minindex += search_result
        refest_match = est_off_argsort[est_off_minindex:]
        refest_matches[ref_on_argsort[ref_id]] = set(refest_match)

        ref_offset = ref[1, ref_off_argsort[-1 - ref_id]]
        est_on_sorted = est[0, est_on_argsort[: (1 + est_on_maxindex)]]
        search_result = np.searchsorted(est_on_sorted, ref_offset, side="right")
        est_on_maxindex = search_result - 1
        estref_match = est_on_argsort[: (1 + est_on_maxindex)]
        estref_matches[ref_off_argsort[-1 - ref_id]] = set(estref_match)

    zip_iterator = zip(refest_matches, estref_matches)
    matches = [x.intersection(y) for (x, y) in zip_iterator]
    return matches


def iou(ref, est, method="fast"):
    """Compute pairwise "intersection over union" (IOU) metric between reference
    events and estimated events.

    Let us denote by a_i and b_i the onset and offset of reference event i.
    Let us denote by u_j and v_j the onset and offset of estimated event j.

    The IOU between events i and j is defined as
        (min(b_i, v_j)-max(a_i, u_j)) / (max(b_i, v_j)-min(a_i, u_j))
    if the events are non-disjoint, and equal to zero otherwise.

    Parameters
    ----------
    ref: np.ndarray [shape=(2, n)], real-valued
         Array of reference events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.

    est: np.ndarray [shape=(2, m)], real-valued
         Array of estimated events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.

    method: str, optional.
         If "fast" (default), computes pairwise intersections via a custom
         dynamic programming algorithm, see fast_intersect.
         If "slow", computes pairwise intersections via bruteforce quadratic
         search, see slow_intersect.

    Returns
    -------
    S: scipy.sparse.dok.dok_matrix, real-valued
        Sparse 2-D matrix. S[i,j] contains the IOU between ref[i] and est[j]
        if these events are non-disjoint and zero otherwise.
    """
    n_refs = ref.shape[1]
    n_ests = est.shape[1]
    S = scipy.sparse.dok_matrix((n_refs, n_ests))

    if method == "fast":
        matches = fast_intersect(ref, est)
    elif method == "slow":
        matches = slow_intersect(ref, est)

    for ref_id in range(n_refs):
        matching_ests = matches[ref_id]
        ref_on = ref[0, ref_id]
        ref_off = ref[1, ref_id]

        for matching_est_id in matching_ests:
            est_on = est[0, matching_est_id]
            est_off = est[1, matching_est_id]
            intersection = min(ref_off, est_off) - max(ref_on, est_on)
            union = max(ref_off, est_off) - min(ref_on, est_on)
            intersection_over_union = intersection / union
            S[ref_id, matching_est_id] = intersection_over_union

    return S


def match_events(ref, est, min_iou=0.0, method="fast"):
    """
    Compute a maximum matching between reference and estimated event times,
    subject to a criterion of minimum intersection-over-union (IOU).

    Given two lists of events ``ref`` (reference) and ``est`` (estimated),
    we seek the largest set of correspondences ``(ref[i], est[j])`` such that
        ``iou(ref[i], est[j]) <= min_iou``
    and such that each ``ref[i]`` and ``est[j]`` is matched at most once.

    This function is strongly inspired by mir_eval.onset.util.match_events.
    It relies on mir_eval's implementation of the Hopcroft-Karp algorithm from
    maximum bipartite graph matching. However, one important difference is that
    mir_eval's distance function relies purely on onset times, whereas this function
    considers both onset times and offset times to compute the IOU metric between
    reference events and estimated events.

    Parameters
    ----------
    ref: np.ndarray [shape=(2, n)], real-valued
         Array of reference events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.

    est: np.ndarray [shape=(2, m)], real-valued
         Array of estimated events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.

    min_iou: real number in [0, 1). Default: 0.
         Threshold for minimum amount of intersection over union (IOU) to match
         any two events. See the iou method for implementation details.

    method: str, optional.
         If "fast" (default), computes pairwise intersections via a custom
         dynamic programming algorithm, see fast_intersect.
         If "slow", computes pairwise intersections via bruteforce quadratic
         search, see slow_intersect.

    Returns
    -------
    matching : list of tuples
        Every tuple corresponds to a match between one reference event and
        one estimated event.
            ``matching[i] == (i, j)`` where ``ref[i]`` matches ``est[j]``.
        Note that all values i and j appear at most once in the list.
    """

    # Intersect reference events and estimated events
    S = iou(ref, est, method=method)

    # Threshold intersection-over-union (IOU) ratio
    S_bool = scipy.sparse.dok_matrix(S > min_iou)
    hits = S_bool.keys()

    # Construct the bipartite graph
    G = {}
    for ref_i, est_i in hits:
        if est_i not in G:
            G[est_i] = []
        G[est_i].append(ref_i)

    # Apply Hopcroft-Karp algorithm (from mir_eval package)
    # to obtain maximum bipartite graph matching
    matching = sorted(mir_eval.util._bipartite_match(G).items())
    return matching


def slow_intersect(ref, est):
    """Find all intersections between reference events and estimated events (slow).
    Best-case complexity: O(N*M) where N=ref.shape[1] and M=est.shape[1]

    Parameters
    ----------
    ref: np.ndarray [shape=(2, n)], real-valued
         Array of reference events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.

    est: np.ndarray [shape=(2, m)], real-valued
         Array of estimated events. Each column is an event.
         The first row denotes onset times and the second row denotes offset times.


    Returns
    -------
    matches: list of sets, length n, integer-valued
         Property: matches[i] contains the set of all indices j such that
            (ref[0, i]<=est[1, j]) AND (ref[1, i]>=est[0, j])
    """
    matches = []
    for i in range(ref.shape[1]):
        matches.append(
            set(
                [
                    j
                    for j in range(est.shape[1])
                    if ((ref[0, i] <= est[1, j]) and (ref[1, i] >= est[0, j]))
                ]
            )
        )
    return matches



MIN_EVAL_VALUE = 0.00001
N_SHOTS = 5
MIN_IOU_TH = 0.3
PRED_FILE_HEADER = ["Audiofilename","Starttime","Endtime"]
POS_VALUE = 'POS'
UNK_VALUE = 'UNK'

def remove_shots_from_ref(ref_df, number_shots=5):
    
    ref_pos_indexes = select_events_with_value(ref_df, value=POS_VALUE)
    ref_n_shot_index = ref_pos_indexes[number_shots-1]
    # remove all events (pos and UNK) that happen before this 5th event
    events_to_drop = ref_df.index[ref_df['Endtime'] <= ref_df.iloc[ref_n_shot_index]['Endtime']].tolist()

    return ref_df.drop(events_to_drop)

def select_events_with_value(data_frame, value=POS_VALUE):

    indexes_list = data_frame.index[data_frame["Q"] == value].tolist()

    return indexes_list

def build_matrix_from_selected_rows(data_frame, selected_indexes_list ):

    matrix_data = np.ones((2, len(selected_indexes_list)))* -1
    for n, idx in enumerate(selected_indexes_list):
        matrix_data[0, n] = data_frame.loc[idx].Starttime # start time for event n
        matrix_data[1, n] = data_frame.loc[idx].Endtime
    return matrix_data


def compute_tp_fp_fn(pred_events_df, ref_events_df):
    # inputs: dataframe with predicted events, dataframe with reference events and their value (POS, UNK, NEG)
    # output: True positives, False Positives, False negatives counts and total number of pos events in ref.

    # makes one pass with bipartite graph matching between pred events and ref positive events
    # get TP
    # make second pass with remaining pred events and ref Unk events
    # compute FP as the number of remaining predicted events after the two rounds of matches.
    # FN is the remaining unmatched pos events in ref.

    ref_pos_indexes = select_events_with_value(ref_events_df, value=POS_VALUE)

    if "Q" not in pred_events_df.columns:
        pred_events_df["Q"] = POS_VALUE
    pred_pos_indexes = select_events_with_value(pred_events_df, value=POS_VALUE)

    ref_1st_round = build_matrix_from_selected_rows(ref_events_df, ref_pos_indexes)
    pred_1st_round = build_matrix_from_selected_rows(pred_events_df, pred_pos_indexes)

    m_pos = match_events(ref_1st_round, pred_1st_round, min_iou=MIN_IOU_TH)
    matched_ref_indexes = [ri for ri, pi in m_pos] 
    matched_pred_indexes = [pi for ri, pi in m_pos]


    ref_unk_indexes = select_events_with_value(ref_events_df, value=UNK_VALUE)
    ref_2nd_round = build_matrix_from_selected_rows(ref_events_df, ref_unk_indexes)

    unmatched_pred_events = list(set(range(pred_1st_round.shape[1])) - set(matched_pred_indexes))
    pred_2nd_round = pred_1st_round[:, unmatched_pred_events]

    m_unk = match_events(ref_2nd_round, pred_2nd_round, min_iou=MIN_IOU_TH)

    # print("# Positive matches between Ref and Pred :", len(m_pos))
    # print("# matches with Unknown events: ", len(m_unk))
    
    tp = len(m_pos)
    fp = pred_1st_round.shape[1] - tp - len(m_unk)
    
    ## compute unmatched pos ref events:
    count_unmached_pos_ref_events = len(ref_pos_indexes) - tp

    fn = count_unmached_pos_ref_events

    total_n_POS_events = len(ref_pos_indexes)
    return tp, fp, fn, total_n_POS_events

def compute_scores_per_class(counts_per_class):

    scores_per_class = {}
    for cl in counts_per_class.keys():
        tp = counts_per_class[cl]["TP"]
        fp = counts_per_class[cl]["FP"]
        fn = counts_per_class[cl]["FN"]

            
        # to compute the harmonic mean we need to have all entries as non zero
        precision = tp/(tp+fp) if tp+fp != 0 else MIN_EVAL_VALUE  # case where no predictions were made 
        if precision < MIN_EVAL_VALUE:
            precision = MIN_EVAL_VALUE
        recall = tp/(fn+tp) if tp != 0 else MIN_EVAL_VALUE
        fmeasure = tp/(tp+0.5*(fp+fn)) if tp != 0 else MIN_EVAL_VALUE

        scores_per_class[cl] = {"precision": precision, "recall": recall, "f-measure": fmeasure}

    return scores_per_class
    
def compute_scores_from_counts(counts):
    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]

    # to compute the harmonic mean we need to have all entries as non zero
    precision = tp/(tp+fp) if tp+fp != 0 else MIN_EVAL_VALUE  # case where no predictions were made 
    if precision < MIN_EVAL_VALUE:
        precision = MIN_EVAL_VALUE 
    recall = tp/(fn+tp) if tp != 0 else MIN_EVAL_VALUE
    fmeasure = tp/(tp+0.5*(fp+fn)) if tp != 0 else MIN_EVAL_VALUE

    scores = {"precision": precision, "recall": recall, "f-measure": fmeasure}
    
    return scores


def build_report(main_set_scores, scores_per_miniset, scores_per_audiofile, save_path, main_set_name="EVAL", **kwargs):
    

    # datetime object containing current date and time
    now = datetime.now()
    date_string = now.strftime("%d%m%Y_%H_%M_%S")
    # print("date and time =", date_string)	

    #make dict:
    report = {
            "set_name": main_set_name,
            "report_date": date_string,
            "overall_scores": main_set_scores,
            "scores_per_subset": scores_per_miniset,
            "scores_per_audiofile": scores_per_audiofile
    }
    if "scores_per_class" in kwargs.keys():
        report["scores_per_class"] = kwargs['scores_per_class']

    with open(os.path.join(save_path,"Evaluation_report_" + "_" + main_set_name + '_' + date_string + '.json'), 'w') as outfile:
        json.dump(report, outfile)

    return


#Fix inputs, not all of this is actually nessecary!
#Also just fix so that this code does not need the og filestructure
#An ugly fix is prob ok for now :)
def evaluate(pred_file_path, ref_file_path, dataset, savepath, metadata=[]):

    #read Gt file structure: get subsets and paths for ref csvs make an inverted dictionary with audiofilenames as keys and folder as value
    gt_file_structure = {}
    gt_file_structure[dataset] = {}
    inv_gt_file_structure = {}
    list_of_subsets = os.listdir(ref_file_path)
    for subset in list_of_subsets:
        gt_file_structure[dataset][subset] = [os.path.basename(fl)[0:-4]+'.wav' for fl in glob(os.path.join(ref_file_path,subset,"*.csv"))]
        for audiofile in gt_file_structure[dataset][subset]:
            inv_gt_file_structure[audiofile] = subset


    #read prediction csv
    pred_csv = pd.read_csv(pred_file_path, dtype=str)
    #verify headers:
    if list(pred_csv.columns) !=  PRED_FILE_HEADER:
        print('Please correct the header of the prediction file. This should be', PRED_FILE_HEADER)
        exit(1)
    #  parse prediction csv
    #  split file into lists of events for the same audiofile.
    pred_events_by_audiofile = dict(tuple(pred_csv.groupby('Audiofilename')))

    counts_per_audiofile = {}
    for audiofilename in list(pred_events_by_audiofile.keys()):
       
               
        # for each audiofile, load correcponding GT File (audiofilename.csv)
        ref_events_this_audiofile_all = pd.read_csv(os.path.join(ref_file_path, inv_gt_file_structure[audiofilename], audiofilename[0:-4]+'.csv'), dtype={'Starttime':np.float64, 'Endtime': np.float64})
        
        #Remove the 5 shots from GT:
        ref_events_this_audiofile = remove_shots_from_ref(ref_events_this_audiofile_all, number_shots=N_SHOTS)
        
        # compare and get counts: TP, FP .. 
        tp_count, fp_count, fn_count , total_n_events_in_audiofile = compute_tp_fp_fn(pred_events_by_audiofile[audiofilename], ref_events_this_audiofile )

        counts_per_audiofile[audiofilename] = {"TP": tp_count, "FP": fp_count, "FN": fn_count, "total_n_pos_events": total_n_events_in_audiofile}
        print(audiofilename, counts_per_audiofile[audiofilename])

    if metadata:
        # using the key for classes => audiofiles,  # load sets metadata:
        with open(metadata) as metadatafile:
                dataset_metadata = json.load(metadatafile)
    else:
        dataset_metadata = copy.deepcopy(gt_file_structure)

    # include audiofiles for which there were no predictions:
    list_all_audiofiles = []
    for miniset in dataset_metadata[dataset].keys():
        if metadata:
            for cl in dataset_metadata[dataset][miniset].keys():
                list_all_audiofiles.extend(dataset_metadata[dataset][miniset][cl] )
        else:
            list_all_audiofiles.extend(dataset_metadata[dataset][miniset])

    for audiofilename in list_all_audiofiles:
        if audiofilename not in counts_per_audiofile.keys():
            ref_events_this_audiofile = pd.read_csv(os.path.join(ref_file_path, inv_gt_file_structure[audiofilename], audiofilename[0:-4]+'.csv'), dtype=str)
            total_n_pos_events_in_audiofile =  len(select_events_with_value(ref_events_this_audiofile, value=POS_VALUE))
            counts_per_audiofile[audiofilename] = {"TP": 0, "FP": 0, "FN": total_n_pos_events_in_audiofile, "total_n_pos_events": total_n_pos_events_in_audiofile}
    


        
    # aggregate the counts per class or subset: 
    list_sets_in_mainset = list(dataset_metadata[dataset].keys())

    counts_per_class_per_set = {}
    scores_per_class_per_set = {}
    counts_per_set = {}
    scores_per_set = {}
    scores_per_audiofile = {}
    for data_set in list_sets_in_mainset:
        # print(data_set)
        
        if metadata:
            list_classes_in_set = list(dataset_metadata[dataset][data_set].keys())

            counts_per_class_per_set[data_set] = {}
            tp_set = 0
            fn_set = 0
            fp_set = 0
            total_n_events_set = 0
            for cl in list_classes_in_set:
                # print(cl)
                list_audiofiles_this_class = dataset_metadata[dataset][data_set][cl]
                tp = 0
                fn = 0
                fp = 0
                total_n_pos_events_this_class = 0
                for audiofile in list_audiofiles_this_class:
                    scores_per_audiofile[audiofile] = compute_scores_from_counts(counts_per_audiofile[audiofile])

                    tp = tp + counts_per_audiofile[audiofile]["TP"]
                    tp_set = tp_set + counts_per_audiofile[audiofile]["TP"]
                    fn = fn + counts_per_audiofile[audiofile]["FN"]
                    fn_set = fn_set + counts_per_audiofile[audiofile]["FN"]
                    fp = fp + counts_per_audiofile[audiofile]["FP"]
                    fp_set = fp_set + counts_per_audiofile[audiofile]["FP"]
                    total_n_pos_events_this_class = total_n_pos_events_this_class + counts_per_audiofile[audiofile]["total_n_pos_events"]
                    total_n_events_set = total_n_events_set + counts_per_audiofile[audiofile]["total_n_pos_events"]
                
                # counts_per_class[cl] = {"TP":tp, "FN": fn, "FP": fp, "total_n_pos_events_this_class": total_n_pos_events_this_class}
                counts_per_class_per_set[data_set][cl] = {"TP": tp, "FN": fn, "FP": fp, "total_n_pos_events_this_class": total_n_pos_events_this_class}
                counts_per_set[data_set] = {"TP": tp_set, "FN": fn_set, "FP": fp_set, "total_n_pos_events_this_set": total_n_events_set}
            
            #  compute scores per subset.  
            scores_per_set[data_set] = compute_scores_from_counts(counts_per_set[data_set])
            #  compute scores per class
            scores_per_class_per_set[data_set] = compute_scores_per_class(counts_per_class_per_set[data_set])  
            
        
        else:
            list_audiofiles_in_set = dataset_metadata[dataset][data_set]
            tp = 0
            fn = 0
            fp = 0
            total_n_pos_events_this_set = 0
            for audiofile in  list_audiofiles_in_set:

                scores_per_audiofile[audiofile] = compute_scores_from_counts(counts_per_audiofile[audiofile])
                tp = tp + counts_per_audiofile[audiofile]["TP"]
                fn = fn + counts_per_audiofile[audiofile]["FN"]
                fp = fp + counts_per_audiofile[audiofile]["FP"]
                total_n_pos_events_this_set = total_n_pos_events_this_set + counts_per_audiofile[audiofile]["total_n_pos_events"]
                counts_per_set[data_set] = {"TP": tp, "FN": fn, "FP": fp, "total_n_pos_events_this_set": total_n_pos_events_this_set}
            
            #  compute scores per subset
            scores_per_set[data_set] = compute_scores_from_counts(counts_per_set[data_set])
                    
    overall_scores = {"precision" : stats.hmean([scores_per_set[dt]["precision"] for dt in scores_per_set.keys()]), 
                    "recall":  stats.hmean([scores_per_set[dt]["recall"] for dt in scores_per_set.keys()]) ,
                    "fmeasure (percentage)": np.round(stats.hmean([scores_per_set[dt]["f-measure"] for dt in scores_per_set.keys()])*100, 3)
                    }
    
    print("\nOverall_scores:",  overall_scores)
    print("\nwriting report")
    if metadata:
        build_report(overall_scores, scores_per_set, scores_per_audiofile,
                savepath, 
                dataset,
                scores_per_class=scores_per_class_per_set)
    else:
        build_report(overall_scores, scores_per_set, scores_per_audiofile,
                savepath, 
                dataset)
    
    return overall_scores

