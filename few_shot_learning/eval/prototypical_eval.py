
import torch
import numpy as np
import os
import pandas as pd
from glob import glob
import h5py
import csv
import datasets.eval_prototypes_base as eval_base

'''
This file requires additional work I feel.
Why not pack pre-processing and actual metric into here.
Spit out eval.csv -> ppeval.csv -> score
Tensorboard the score?
How to pass writer around?
'''

#Do we like this relative import?
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
    
def eval_TrainValCV(model, test_loader, config, writer, fold, class_map, class_dict, tr_cls_keys, val_cls_keys):
    return eval_help_TrainValCV(None, None, config, writer, class_map, class_dict, tr_cls_keys, val_cls_keys, fold, 'TEST')
    
    
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
        if config.experiment.eval.base_eval:
            onset, offset = eval_base.evaluate_prototypes(config, hdf_eval, device, strt_index_query, model)
        else:
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
        util.post_processing(csv_path, 'PP_'+csv_path, tag, config)
        scores = util.evaluate('PP_'+csv_path, config.experiment.path.val_OG, tag, './')
        #How the fuck are we getting precision and recall on the validation data?
        #Shoudln't that be done here like the test?
        #So we call this function with tag val from the training loop
        #And since scores is returned at the bottom we have access to the data and it is there added to the writer.
        
    elif tag == 'TEST':
        csv_path = 'TEST_out.csv'
        df_out.to_csv(csv_path,index=False)
        util.post_processing(csv_path, 'PP_'+csv_path, tag, config)
        scores = util.evaluate('PP_'+csv_path, config.experiment.path.test_OG, tag, './')
        if writer is not None:
            
            writer.add_scalar('Fmeasure/test', scores['fmeasure (percentage)'])
            writer.add_scalar('precision/test', scores['precision'])
            writer.add_scalar('recall/test', scores['recall'])
    
    return scores
    
'''
TODO: Work with the tag so we can do both VAL and TEST here as in eval_help
Important that it actually is tr_cls_keys here and NOT val
'''
def eval_help_TrainValCV(model, test_loader, config, writer, class_map, class_dict, tr_cls_keys, val_cls_keys, fold, tag):
    
    if config.experiment.set.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    name_arr = np.array([])
    onset_arr = np.array([])
    offset_arr = np.array([])
    
    if tag == 'VAL':
    
        for key in val_cls_keys:
            for inner in class_map[key]:
                #Probably do a check here if there are five shots etc...
                if len(class_dict[inner]['start_pos']) < config.experiment.train.n_shot:
                    continue
                f_path = os.path.join(config.experiment.path.trainvalcv_features, class_dict[inner]['file_path'].replace('.csv', '.h5').split('/')[-1])
                h_file = h5py.File(f_path, 'r')
                #TODO: This is a bug!!!!
                #start_index_query is not right here. Just check what we to in datagen for example.
                strt_index_query = h_file[inner+'_start_index_query'][:][0]
                #We pase the solution done in datagen here and see how that works. This might be wrong to we should really calculate this correctly!!!!
                
                if config.experiment.eval.base_eval:
                    onset, offset = eval_base.evaluate_prototypes(config, h_file, device, strt_index_query, model, class_map, class_dict, inner, tr_cls_keys)
                else:
                    onset,offset = util.evaluate_prototypes(config, h_file, device, strt_index_query, model, class_map, class_dict, inner, tr_cls_keys)

                #Might need to add +'.wav' to inner here to conform to existing code
                name = np.repeat(inner+'.wav', len(onset))
                name_arr = np.append(name_arr,name)
                onset_arr = np.append(onset_arr,onset)
                offset_arr = np.append(offset_arr,offset)
                
    elif tag == 'TEST':
        
        all_feat_files = [file for file in glob(os.path.join(config.experiment.path.test_features,'*.h5'))]
        for feat_file in all_feat_files:
            feat_name = feat_file.split('/')[-1]
            audio_name = feat_name.replace('h5','wav')

            print("Processing audio file : {}".format(audio_name))

            hdf_eval = h5py.File(feat_file,'r')
            strt_index_query =  hdf_eval['start_index_query'][:][0]
            if config.experiment.eval.base_eval:
                onset, offset = eval_base.evaluate_prototypes(config, hdf_eval, device, strt_index_query, model)
            else:
                onset,offset = util.evaluate_prototypes(config, hdf_eval, device, strt_index_query, model)


            name = np.repeat(audio_name,len(onset))
            name_arr = np.append(name_arr,name)
            onset_arr = np.append(onset_arr,onset)
            offset_arr = np.append(offset_arr,offset)
    
    df_out_pred = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
    
    if tag == 'VAL':
        csv_path = 'VAL_out_fold_'+str(fold)+'.csv'
        #df_out.to_csv(csv_path,index=False)
        #util.post_processing(csv_path, 'PP_'+csv_path, tag, config)
        #Here we need to do a new 'val_OG' then we good to go I think.
        #Use info in class_dict for this

        if not os.path.exists(os.path.join(os.getcwd(), 'VAL_folds')):
            os.mkdir('VAL_folds')
        if not os.path.exists(os.path.join(os.getcwd(), 'VAL_folds/fold_'+str(fold))):
            os.mkdir('VAL_folds/fold_'+str(fold))
        if not os.path.exists(os.path.join(os.getcwd(), 'VAL_folds/fold_'+str(fold)+'/fold')):
            os.mkdir('VAL_folds/fold_'+str(fold)+'/fold')
            
        for key in val_cls_keys:
            for inner in class_map[key]:
                if len(class_dict[inner]['start_pos']) < config.experiment.train.n_shot:
                    continue
                #f_csv = open('VAL_folds/fold_'+str(fold)+'/'+inner+'.csv', 'w')
                f_wav = open('VAL_folds/fold_'+str(fold)+'/'+'/fold/'+inner+'.wav', 'w')

                #Print csv file. Look at original files for format.
                #Do POS and UNK come in order, don't think so?
                name_arr = np.array([])
                onset_arr = np.array([])
                offset_arr = np.array([])
                q_arr = np.array([])

                onset_arr = np.append(onset_arr, class_dict[inner]['start_pos'])
                offset_arr = np.append(offset_arr, class_dict[inner]['end_pos'])
                q_arr = np.append(q_arr, ['POS']*len(onset_arr))

                tmp_start_unk = class_dict[inner]['start_unk']
                onset_arr = np.append(onset_arr, tmp_start_unk)
                offset_arr = np.append(offset_arr, class_dict[inner]['end_unk'])
                q_arr = np.append(q_arr, ['UNK']*len(tmp_start_unk))

                name_arr = np.append(name_arr, [inner+'.csv']*len(onset_arr))
                df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr, 'Q':q_arr})
                df_out.to_csv('VAL_folds/fold_'+str(fold)+'/'+'/fold/'+inner+'.csv', index=False)

                #f_csv.close()
                f_wav.close()
                
        df_out_pred.to_csv(csv_path, index=False)
        util.post_processing(csv_path, 'PP_'+csv_path, tag, config, fold)
        scores = util.evaluate('PP_'+csv_path, 'VAL_folds/fold_'+str(fold), tag, './')
        
    elif tag == 'TEST':
        csv_path = 'TEST_out.csv'
        df_out_pred.to_csv(csv_path,index=False)
        util.post_processing(csv_path, 'PP_'+csv_path, tag, config)
        scores = util.evaluate('PP_'+csv_path, config.experiment.path.test_OG, tag, './')
        if writer is not None:
            
            writer.add_scalar('Fmeasure/test', scores['fmeasure (percentage)'])
            writer.add_scalar('precision/test', scores['precision'])
            writer.add_scalar('recall/test', scores['recall'])
    
    return scores



def load():
    return eval

def load_TrainValCV():
    return eval_TrainValCV

def load_help():
    return eval_help

def load_help_TrainValCV():
    return eval_help_TrainValCV