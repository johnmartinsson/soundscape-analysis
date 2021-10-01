import abc
import os
import librosa
from glob import glob
import numpy as np
import h5py
from itertools import chain
import pandas as pd
import datasets.dcase_few_shot_bioacoustic as util

class FeatureExtractor(abc.ABC):
    
    def __init__(self):
        pass

class SpectralFeatureExtractor(FeatureExtractor):

    def __init__(self, config, spectralizer):
        self.config = config
        self.spectralizer = spectralizer

    def extract_features(self):
        
        self.extract_train()
        self.extract_test()

    def extract_train(self):

        print('--- Processing training data ---')
        csv_files = [file for file in glob(os.path.join(self.config.path.data_train, '*.csv'))]

        fps = self.config.features.sr / self.config.features.hop_mel
        seg_len = int(round(self.config.features.seg_len * fps))
        hop_seg = int(round(self.config.features.hop_seg * fps))
        
        labels = []
        events = []
        
        for file in csv_files:
            
            print('Processing ' + file.replace('csv', 'wav'))
            audio, sr = librosa.load(file.replace('csv', 'wav'), self.config.features.sr)
            
            print('Spectral transform')
            pcen = self.spectralizer.raw_to_spec(audio, self.config)
            
            df = pd.read_csv(file, header=0, index_col=False)
            df_pos = df[(df == 'POS').any(axis=1)]
            
            start_time, end_time = util.time_2_frame(df_pos, fps)
            label_f = list(chain.from_iterable(
                [df_pos.columns[(df_pos == 'POS').loc[index]].values for index, _ in df_pos.iterrows()]))
            
            print('Slicing spectrogram')
            
            for index in range(len(start_time)):
                
                str_ind = start_time[index]
                end_ind = end_time[index]
                label = label_f[index]
                
                #Event longer than a segment?
                if end_ind - str_ind > seg_len:
                    shift = 0
                    while end_ind - (str_ind + shift) > seg_len:
                        
                        pcen_patch = pcen[int(str_ind + shift):int(str_ind + shift + seg_len)]
                        events += [pcen_patch]
                        labels.append(label)
                        shift += hop_seg
                    
                    pcen_patch = pcen[end_ind - seg_len:end_ind]
                    events += [pcen_patch]
                    labels.append(label)
                
                #Event shorter than a segment!
                else:
                    
                    #Repeat the patch til segment length.
                    pcen_patch = pcen[str_ind:end_ind]
                    if pcen_patch.shape[0] == 0:
                        continue
                    
                    repeats = int(seg_len/(pcen_patch.shape[0])) + 1
                    pcen_patch_new = np.tile(pcen_patch, (repeats, 1))
                    pcen_patch_new = pcen_patch_new[0:int(seg_len)]
                    events += [pcen_patch_new]
                    labels.append(label)
                    
        print('Writing to file')
        
        events = np.array(events)
        
        hf = h5py.File(os.path.join(self.config.path.train_w, 'mel_train.h5'), 'w')
        hf.create_dataset('features', data=events)
        hf.create_dataset('labels', data=[s.encode() for s in labels], dtype='S20')
        hf.close()
        
        print('Done')

    def extract_test(self):

        print('--- Processing test data ---')
        csv_files = [file for file in glob(os.path.join(self.config.path.data_test, '*.csv'))]

        fps = self.config.features.sr / self.config.features.hop_mel
        seg_len = int(round(self.config.features.seg_len * fps))
        hop_seg = int(round(self.config.features.hop_seg * fps))

        for file in csv_files:
                
            print('Processing ' + file.replace('csv', 'wav'))
            
            idx_pos = 0
            idx_neg = 0
            start_neg = 0
            hop_neg = 0
            idx_query = 0
            hop_query = 0
            strt_index = 0

            split_list = file.split('/')
            name = str(split_list[-1].split('.')[0])
            feat_name = name + '.h5'
            audio_path = file.replace('csv', 'wav')
            feat_info = []
            hdf_eval = os.path.join(self.config.path.test_w ,feat_name)
            hf = h5py.File(hdf_eval,'w')
            hf.create_dataset('feat_pos', shape=(0, seg_len, self.config.features.n_mels),
                                maxshape= (None, seg_len, self.config.features.n_mels))
            hf.create_dataset('feat_query',shape=(0,seg_len, self.config.features.n_mels),maxshape=(None,seg_len,self.config.features.n_mels))
            hf.create_dataset('feat_neg',shape=(0,seg_len, self.config.features.n_mels),maxshape=(None,seg_len,self.config.features.n_mels))
            hf.create_dataset('start_index_query',shape=(1,),maxshape=(None))

            'In case you want to use the statistics of each file to normalize'

            hf.create_dataset('mean_global',shape=(1,), maxshape=(None))
            hf.create_dataset('std_dev_global',shape=(1,), maxshape=(None))

            df_eval = pd.read_csv(file, header=0, index_col=False)
            Q_list = df_eval['Q'].to_numpy()

            start_time,end_time = util.time_2_frame(df_eval,fps)

            index_sup = np.where(Q_list == 'POS')[0][:self.config.train.n_shot]

            audio, sr = librosa.load(file.replace('csv', 'wav'), self.config.features.sr)
            print('Spectral transform')
            pcen = self.spectralizer.raw_to_spec(audio, self.config)
            
            mean = np.mean(pcen)
            std = np.mean(pcen)
            hf['mean_global'][:] = mean
            hf['std_dev_global'][:] = std

            strt_indx_query = end_time[index_sup[-1]]
            end_idx_neg = pcen.shape[0] - 1
            hf['start_index_query'][:] = strt_indx_query

            print("Creating negative dataset")

            while end_idx_neg - (strt_index + hop_neg) > seg_len:

                patch_neg = pcen[int(strt_index + hop_neg):int(strt_index + hop_neg + seg_len)]

                hf['feat_neg'].resize((idx_neg + 1, patch_neg.shape[0], patch_neg.shape[1]))
                hf['feat_neg'][idx_neg] = patch_neg
                idx_neg += 1
                hop_neg += hop_seg

            last_patch = pcen[end_idx_neg - seg_len:end_idx_neg]
            hf['feat_neg'].resize((idx_neg + 1, last_patch.shape[0], last_patch.shape[1]))
            hf['feat_neg'][idx_neg] = last_patch

            print("Creating Positive dataset")
            for index in index_sup:

                str_ind = int(start_time[index])
                end_ind = int(end_time[index])

                if end_ind - str_ind > seg_len:

                    shift = 0
                    while end_ind - (str_ind + shift) > seg_len:

                        patch_pos = pcen[int(str_ind + shift):int(str_ind + shift + seg_len)]

                        hf['feat_pos'].resize((idx_pos + 1, patch_pos.shape[0], patch_pos.shape[1]))
                        hf['feat_pos'][idx_pos] = patch_pos
                        idx_pos += 1
                        shift += hop_seg
                    last_patch_pos = pcen[end_ind - seg_len:end_ind]
                    hf['feat_pos'].resize((idx_pos + 1, patch_pos.shape[0], patch_pos.shape[1]))
                    hf['feat_pos'][idx_pos] = last_patch_pos
                    idx_pos += 1

                else:
                    patch_pos = pcen[str_ind:end_ind]

                    if patch_pos.shape[0] == 0:
                        print(patch_pos.shape[0])
                        print("The patch is of 0 length")
                        continue
                    repeat_num = int(seg_len / (patch_pos.shape[0])) + 1

                    patch_new = np.tile(patch_pos, (repeat_num, 1))
                    patch_new = patch_new[0:int(seg_len)]
                    hf['feat_pos'].resize((idx_pos + 1, patch_new.shape[0], patch_new.shape[1]))
                    hf['feat_pos'][idx_pos] = patch_new
                    idx_pos += 1



            print("Creating query dataset")

            while end_idx_neg - (strt_indx_query + hop_query) > seg_len:

                patch_query = pcen[int(strt_indx_query + hop_query):int(strt_indx_query + hop_query + seg_len)]
                hf['feat_query'].resize((idx_query + 1, patch_query.shape[0], patch_query.shape[1]))
                hf['feat_query'][idx_query] = patch_query
                idx_query += 1
                hop_query += hop_seg


            last_patch_query = pcen[end_idx_neg - seg_len:end_idx_neg]

            hf['feat_query'].resize((idx_query + 1, last_patch_query.shape[0], last_patch_query.shape[1]))
            hf['feat_query'][idx_query] = last_patch_query

            hf.close()




class RawFeatureExtractor(FeatureExtractor):

    def __init__(self, config):
        self.config = config


    def extract_features(self):
        self.extract_train()
        self.extract_test()

    def extract_train(self):

        print('--- Processing training data ---')
        csv_files = [file for file in glob(os.path.join(self.config.path.data_train, '*.csv'))]

        events = []
        labels = []
        
        for file in csv_files:
        
            print('Processing ' + file.replace('csv', 'wav'))
            audio, sr = librosa.load(file.replace('csv', 'wav'), self.config.features.sr)
            df = pd.read_csv(file, header=0, index_col=False)
            df_pos = df[(df == 'POS').any(axis=1)]
            
            #Add config options for window size around event
            df_pos.loc[:, 'Starttime'] = df_pos['Starttime'] - 0.025
            df_pos.loc[:, 'Endtime'] = df_pos['Endtime'] + 0.025
            start_time = [int(np.floor(start * sr)) for start in df_pos['Starttime']]
            end_time = [int(np.floor(end * sr)) for end in df_pos['Endtime']]
            
            #Better way of doing this?
            for i in range(len(start_time)):
                events += [audio[start_time[i]:end_time[i]]]
                
            labels += list(chain.from_iterable(
                [df_pos.columns[(df_pos == 'POS').loc[index]].values for index, _ in df_pos.iterrows()]))
        
        print('Padding')
        #Pad arrays in events and format for write
        max_len = 0
        for e in events:
            if len(e) > max_len:
                max_len = len(e)
                
        for i in range(len(events)):
            if len(events[i]) < max_len:
                events[i] = np.append(events[i], np.array([self.config.features.raw_pad]*(max_len-len(events[i]))))
        
        events = np.array(events)
        
        print('Writing to file')
        
        hf = h5py.File(os.path.join(self.config.path.train_w, 'raw_train.h5'), 'w')
        hf.create_dataset('features', data=events)
        hf.create_dataset('labels', data=[s.encode() for s in labels], dtype='S20')
        hf.close()
        
        print('Done')

    #TODO
    def extract_test(self):
        pass

#Dont know where else to put this right now.
#This class could for sure be of use in many SED tasks
class Spectralizer():
    
    def __init__(self, config):
        self.config = config
        
        self.sr = config.features.sr
        self.n_fft = config.features.n_fft
        self.hop = config.features.hop_mel
        self.n_mels = config.features.n_mels
        self.fmax = config.features.fmax
        

    def raw_to_spec(self, audio, config):

        #Supposedly suggested by librosa.
        audio = audio * (2**32)

        mel_spec = librosa.feature.melspectrogram(audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop,
                                                 n_mels=self.n_mels, fmax=self.fmax)

        pcen = librosa.core.pcen(mel_spec, sr=self.sr)
        pcen = pcen.astype(np.float32)
        
        #Note that we transform the features here and therefor have time/frame along dim 0.
        #Transform back when loading data? Smaksak
        return pcen.T