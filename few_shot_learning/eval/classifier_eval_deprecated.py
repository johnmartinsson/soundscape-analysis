'''
Borrow heavily from prot eval. We basically can reuse the code but just use the embedding output from the model
'''


import torch
import numpy as np
import os
import pandas as pd
from glob import glob
import h5py
import csv

#Do we like this relative import?

'''
2021/11/29:
At the current moment I can't see what use we have of this file?
There is literally no difference between this and prototypical eval.
Good practice is to not use this anymore. Rename file for now.
'''



import datasets.dcase_few_shot_bioacoustic as util

#TODO: Have writer add scalar for fmeasure/precision/recall.
def eval(model, test_loader, config, writer):
    '''
    We can work from the assumption that if this function is called it is from a test scenario
    So we always call help with the tag TEST and model = None so that we load the best one.
    '''
    return eval_help(None, None, config, writer, 'TEST')
    '''
    if config.experiment.eval.dataset == 'VAL':
        return eval_help(model, test_loader, config, writer, 'VAL')
    elif config.experiment.eval.dataset == 'TEST':
        return eval_help(model, test_loader, config, writer, 'TEST')
    elif config.experiment.eval.dataset == 'VALTEST':
        #Perhaps no need to return these?
        #We most likely will do these one by one for the neg exp
        eval_help(model, test_loader, config, writer, 'VAL')
        eval_help(model, test_loader, config, writer, 'TEST')
    '''
def eval_help(model, test_loader, config, writer, tag):
    
    if config.experiment.set.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    name_arr = np.array([])
    onset_arr = np.array([])
    offset_arr = np.array([])
    
    if tag == 'VAL':
        all_feat_files = [file for file in glob(os.path.join(config.experiment.path.val_features,'*.h5'))]
    elif tag == 'TEST':
        all_feat_files = [file for file in glob(os.path.join(config.experiment.path.test_features,'*.h5'))]

    for feat_file in all_feat_files:
        feat_name = feat_file.split('/')[-1]
        audio_name = feat_name.replace('h5','wav')

        print("Processing audio file : {}".format(audio_name))

        hdf_eval = h5py.File(feat_file,'r')
        strt_index_query =  hdf_eval['start_index_query'][:][0]
        onset,offset = util.evaluate_prototypes(config, hdf_eval, device, strt_index_query, model)

        name = np.repeat(audio_name,len(onset))
        name_arr = np.append(name_arr,name)
        onset_arr = np.append(onset_arr,onset)
        offset_arr = np.append(offset_arr,offset)

    df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
    #csv_path = os.path.join(config.experiment.root_path,'Eval_out.csv')
    #Do this instead for now, configurable best in the end.
    
    #Eval pipe!
    if tag == 'VAL':
        '''
        Perhaps we would like to make it possible to save all of the validation results and not just have them
        available for the last epoch, TODO
        '''
        csv_path = 'VAL_out.csv'
        df_out.to_csv(csv_path,index=False)
        util.post_processing(config.experiment.path.data_val, csv_path, 'PP_'+csv_path)
        scores = util.evaluate('PP_'+csv_path, config.experiment.path.val_OG, tag, './')
        
    elif tag == 'TEST':
        csv_path = 'TEST_out.csv'
        df_out.to_csv(csv_path,index=False)
        util.post_processing(config.experiment.path.data_test, csv_path, 'PP_'+csv_path)
        scores = util.evaluate('PP_'+csv_path, config.experiment.path.test_OG, tag, './')
        if writer is not None:
            writer.add_scalar(tag+'_fmeasure', scores['fmeasure (percentage)'])
            writer.add_scalar(tag+'precision', scores['precision'])
            writer.add_scalar(tag+'recall', scores['recall'])
    
    return scores
    
'''
Why is this block of code even here?

def dummy_choice(csv, n_shots):
    events = []
    for i in range(len(csv)):
                if(csv.loc[i].values[-1] == 'POS' and len(events) < n_shots):
                    events.append(csv.loc[i].values)
    return events

#Might wanna check the paths here and if we are please with the output.
def post_processing(val_path, evaluation_file, new_evaluation_file, n_shots=5):
    
    
    csv_files = [file for file in glob(os.path.join(val_path, '*.csv'))]
    
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

'''
    
    
def load():
    return eval

def load_help():
    return eval_help