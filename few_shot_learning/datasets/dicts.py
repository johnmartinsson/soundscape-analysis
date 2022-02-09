import os
from glob import glob
from collections import defaultdict
import pandas as pd

def get_dicts(config):
    
    train_path = config.experiment.path.data_train
    val_path = config.experiment.path.data_val

    train_csv = [file for file in glob(os.path.join(train_path, '*.csv'))]
    val_csv = [file for file in glob(os.path.join(val_path, '*.csv'))]
    all_csv = train_csv + val_csv
    
    CLS_START_INDEX = 3

    class_map = defaultdict(list)
    class_dict = {}

    for csv_file in train_csv:

        df = pd.read_csv(csv_file)
        num_classes = len(df.columns[3:])

        for i in range(num_classes):

            class_name = csv_file.split('/')[-1].split('.')[0]+'_'+str(i)
            class_map[df.columns[CLS_START_INDEX+i]].append(class_name)
            cols = list(range(CLS_START_INDEX)) + [i+CLS_START_INDEX]
            sub_df = df.iloc[:, cols]
            sub_df_POS = sub_df[(sub_df == 'POS').any(axis=1)]
            starttime_pos = sub_df_POS['Starttime'].values
            endtime_pos = sub_df_POS['Endtime'].values
            sub_df_UNK = sub_df[(sub_df == 'UNK').any(axis=1)]
            starttime_unk = sub_df_UNK['Starttime'].values
            endtime_unk = sub_df_UNK['Endtime'].values

            '''
            if len(starttime_pos) >= config.train.sample_threshold:
                class_dict[class_name] = {'file_path': csv_file, 'start_pos' : starttime_pos, 'end_pos' : endtime_pos, 'start_unk' : starttime_unk, 'end_unk' : endtime_unk}
            '''
            class_dict[class_name] = {'file_path': csv_file, 'start_pos' : starttime_pos, 'end_pos' : endtime_pos, 'start_unk' : starttime_unk, 'end_unk' : endtime_unk}

    for csv_file in val_csv:

        df = pd.read_csv(csv_file)
        num_classes = len(df.columns[3:])

        for i in range(num_classes):

            class_name = csv_file.split('/')[-1].split('.')[0]+'_'+str(i)
            class_map[class_name].append(class_name)
            cols = list(range(CLS_START_INDEX)) + [i+CLS_START_INDEX]
            sub_df = df.iloc[:, cols]
            sub_df_POS = sub_df[(sub_df == 'POS').any(axis=1)]
            starttime_pos = sub_df_POS['Starttime'].values
            endtime_pos = sub_df_POS['Endtime'].values
            sub_df_UNK = sub_df[(sub_df == 'UNK').any(axis=1)]
            starttime_unk = sub_df_UNK['Starttime'].values
            endtime_unk = sub_df_UNK['Endtime'].values

            '''
            if len(starttime_pos) >= config.train.sample_threshold:
                class_dict[class_name] = {'file_path': csv_file, 'start_pos' : starttime_pos, 'end_pos' : endtime_pos, 'start_unk' : starttime_unk, 'end_unk' : endtime_unk}
            '''
            class_dict[class_name] = {'file_path': csv_file, 'start_pos' : starttime_pos, 'end_pos' : endtime_pos, 'start_unk' : starttime_unk, 'end_unk' : endtime_unk}
            
    return class_map, class_dict