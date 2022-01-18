import abc
import os
import librosa
from glob import glob
import numpy as np
import h5py
from itertools import chain
import pandas as pd
import datasets.dcase_few_shot_bioacoustic as util
from tqdm import tqdm

class FeatureExtractor(abc.ABC):
    
    def __init__(self):
        pass

'''
Is this flexible enough? Then again I don't want to change to much. Just makes the rest of the code messy/need changeing.
'''
    
class SpectralFeatureExtractor(FeatureExtractor):

    def __init__(self, config, spectralizer):
        self.config = config
        self.spectralizer = spectralizer
        #Should I have an 'all' file here as to be reachable by all functions?

    def extract_features(self):
        if self.config.set.train:
            self.extract_train()
        if self.config.set.val:
            self.extract_test('VAL')
        if self.config.set.test:
            self.extract_test('TEST')

    def extract_train(self):
        
        '''
        So we would like to extract more or less the whole audio file so we can work negatives into the training procedure.
        Initial thoughts: Keeping the same process for the labeled training data and spitting out a 'mel_train.h5' file as we do now is good.
        Not changing this leads to a whole lot of less changes else where. We might even be interested in just creating another h5 file all togehter instead of 
        adding a new dataset. This could for example be mel_train_neg.h5. There are some considerations which needs to be adressed either way:
            (i) Should each processed file be put into seperate datasets?
            (ii) Should we try to process out the labeled bits of the files?
        
        Update:
        I think this is the way: Create a separate h5 file for each audio file as for the val and test set.
        For each h5 file have one dataset named 'neg' just as for the val and test files.
        This makes it relatively easy to work with.
        In the output folder there will be three folders train, val and test.
        The train folder contains 'mel_train.h5' and a separate h5 file for each audio file as described above.
        Possibly in separate folder to make it even easier to work with.
        Should be relatively easy to work with.
        Don't bother not including annotated segments, should not matter at all really.
        '''

        print('--- Processing training data ---')
        csv_files = [file for file in glob(os.path.join(self.config.path.data_train, '*.csv'))]

        fps = self.config.features.sr / self.config.features.hop_mel
        seg_len = int(round(self.config.features.seg_len * fps))
        hop_seg = int(round(self.config.features.hop_seg * fps))
        
        labels = []
        events = []
        
        for file in tqdm(csv_files):
            
            print('Processing ' + file.replace('csv', 'wav'))
            audio, sr = librosa.load(file.replace('csv', 'wav'), self.config.features.sr)
            
            print('Spectral transform')
            #I wonder if you shouldn't just normalize here to be honest.
            #How do we actually do this well in regards to how the training data is normalized now?
            #Not pleased actually.
            pcen = self.spectralizer.raw_to_spec(audio, self.config)
            
            
            '''
            TODO: We should only extract segments without no annotations or all annotations being NEG.
            At least for use in the semi approach. We could do some statistics on the annotations files to find out as well if it is nessecary.
            
            Create some inverted time_2_frame? Give onset/offset of parts without annotations?
            We increasingly run into the problem of choosing edges though right?
            Is there an easy way of handeling this?
            The best might just to simply use other audio sources. No headache really, if the files are long.
            
            Can we use this method with just empty annotations files?
            '''
            
            if self.config.set.extract_all:
                print('Extracting all segments')
                split_list = file.split('/')
                name = str(split_list[-1].split('.')[0])
                feat_name = name + '.h5'
                file_path = os.path.join(self.config.path.output, 'hfiles/train/whole')
                hf_whole = h5py.File(os.path.join(file_path, feat_name), 'w')
                hf_whole.create_dataset('feat_neg',shape=(0,seg_len, self.config.features.n_mels),maxshape=(None,seg_len,self.config.features.n_mels))
                hf_whole.create_dataset('mean_global',shape=(1,), maxshape=(None))
                hf_whole.create_dataset('std_global',shape=(1,), maxshape=(None))
                strt_index = 0
                idx_neg = 0
                end_idx_neg = pcen.shape[0]-1
                hop_neg = 0
                while end_idx_neg - (strt_index + hop_neg) > seg_len:

                    patch_neg = pcen[int(strt_index + hop_neg):int(strt_index + hop_neg + seg_len)]

                    hf_whole['feat_neg'].resize((idx_neg + 1, patch_neg.shape[0], patch_neg.shape[1]))
                    hf_whole['feat_neg'][idx_neg] = patch_neg
                    idx_neg += 1
                    hop_neg += hop_seg

                last_patch = pcen[end_idx_neg - seg_len:end_idx_neg]
                hf_whole['feat_neg'].resize((idx_neg + 1, last_patch.shape[0], last_patch.shape[1]))
                hf_whole['feat_neg'][idx_neg] = last_patch

                hf_whole['mean_global'][:] = np.mean(pcen)
                hf_whole['std_global'][:] = np.std(pcen)

                hf_whole.close()
            
            df = pd.read_csv(file, header=0, index_col=False)
            df_pos = df[(df == 'POS').any(axis=1)]
            
            start_time, end_time = util.time_2_frame(df_pos, fps)
            
            
            label_f = list(chain.from_iterable(
                [df_pos.columns[(df_pos == 'POS').loc[index]].values for index, _ in df_pos.iterrows()]))
            
            print('Extracting annotated segemnts')
            
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
        
        out_path = os.path.join(self.config.path.output, 'hfiles/train')
        hf = h5py.File(os.path.join(out_path, 'mel_train.h5'), 'w') #TODO
        hf.create_dataset('features', data=events)
        hf.create_dataset('labels', data=[s.encode() for s in labels], dtype='S20')
        hf.close()
        
        print('Done')

    #Make it so that this can handle the official evaluation set as well.
    def extract_test(self, tag):
        
        #TODO, make switch between VAL or TEST possible.
        
        if tag == 'VAL':
            print('--- Processing validation data ---')
            csv_files = [file for file in glob(os.path.join(self.config.path.data_val, '*.csv'))]
        elif tag == 'TEST':
            print('--- Processing test data ---')
            csv_files = [file for file in glob(os.path.join(self.config.path.data_test, '*.csv'))]
        else:
            raise Exception
                
        fps = self.config.features.sr / self.config.features.hop_mel
        seg_len = int(round(self.config.features.seg_len * fps))
        hop_seg = int(round(self.config.features.hop_seg * fps))

        for file in tqdm(csv_files):
                
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
            
            if tag == 'VAL':
                hf_path = os.path.join(self.config.path.output, 'hfiles/val')
                hdf_eval = os.path.join(hf_path ,feat_name)
            elif tag == 'TEST':
                hf_path = os.path.join(self.config.path.output, 'hfiles/test')
                hdf_eval = os.path.join(hf_path ,feat_name)
                
            hf = h5py.File(hdf_eval,'w')
            hf.create_dataset('feat_pos', shape=(0, seg_len, self.config.features.n_mels),
                                maxshape= (None, seg_len, self.config.features.n_mels))
            hf.create_dataset('feat_query',shape=(0,seg_len, self.config.features.n_mels),maxshape=(None,seg_len,self.config.features.n_mels))
            hf.create_dataset('feat_neg',shape=(0,seg_len, self.config.features.n_mels),maxshape=(None,seg_len,self.config.features.n_mels))
            hf.create_dataset('start_index_query',shape=(1,),maxshape=(None))

            'In case you want to use the statistics of each file to normalize'
            '''
            TODO: Should we actually use these instead. Should we normalize per file?
            How to normalize the unlabeled data currently sorted into h5 files.
            Use the datagen objective?
            This warrants a discussion for sure.
            '''

            hf.create_dataset('mean_global',shape=(1,), maxshape=(None))
            hf.create_dataset('std_global',shape=(1,), maxshape=(None))

            df_eval = pd.read_csv(file, header=0, index_col=False)
            Q_list = df_eval['Q'].to_numpy()

            start_time,end_time = util.time_2_frame(df_eval,fps)

            index_sup = np.where(Q_list == 'POS')[0][:self.config.train.n_shot]

            audio, sr = librosa.load(file.replace('csv', 'wav'), self.config.features.sr)
            print('Spectral transform')
            pcen = self.spectralizer.raw_to_spec(audio, self.config)
            
            mean = np.mean(pcen)
            std = np.std(pcen)
            hf['mean_global'][:] = mean
            hf['std_global'][:] = std

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
        csv_files = [file for file in glob(os.path.join(self.config.experiment.path.data_train, '*.csv'))]

        events = []
        labels = []
        
        for file in csv_files:
        
            print('Processing ' + file.replace('csv', 'wav'))
            audio, sr = librosa.load(file.replace('csv', 'wav'), self.config.experiment.features.sr)
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