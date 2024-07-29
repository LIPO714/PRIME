import json
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def read_pkl(path):
    with open(path, 'rb') as file:
        # 使用pickle.load()方法加载对象
        data = pickle.load(file)
    return data

def preprocess(data, cut_time, age_list, feature):
    # dict_keys(['reg_ts', 'name', 'ts_tt', 'irg_ts', 'irg_ts_mask', 'text_data', 'text_time_to_start', 'ihm_label', 'ihm_task', 'los', 'decom_time', 'demographics', 'pheno_label'])
    maxer_512 = 0
    for i in tqdm(range(len(data)), desc="Processing", unit="iteration"):
        now_data = data[i]
        ts_data = now_data['irg_ts']
        ts_mask = now_data['irg_ts_mask']
        ts_tt = np.array(now_data['ts_tt'])

        start_time = ts_tt[0]
        ts_tt = ts_tt - start_time  # 开始时间对齐到0

        if cut_time is not None:
            cut_time = float(cut_time)
            ts_before_num = np.sum(ts_tt <= cut_time + 1e-6)
            ts_data = ts_data[:ts_before_num]
            ts_mask = ts_mask[:ts_before_num]
            ts_tt = ts_tt[:ts_before_num]

        ages = now_data['demographics'][0]
        if ages < 120:
            age_list.append(ages)

        # feature
        ts_data[ts_mask==0] = np.nan
        feature.append(ts_data)

    # print("Maxer 512:", maxer_512)
    return age_list, feature

if __name__ == '__main__':
    train_data_path = '../data/train_ts_note_data.pkl'
    test_data_path = '../data/test_ts_note_data.pkl'
    cut_time = '48'  # None  # '48'
    metadata_path = '../data/metadata.json'

    train_data = read_pkl(train_data_path)
    test_data = read_pkl(test_data_path)

    age_list = []
    feature = []
    tokenizer = AutoTokenizer.from_pretrained('../Clinical-Longformer')  # padding用的是1
    max_token_id = max(tokenizer.get_vocab().values())
    print("max:", max_token_id)

    age_list, feature = preprocess(train_data, cut_time, age_list, feature)
    age_list, feature = preprocess(test_data, cut_time, age_list, feature)

    age_list = np.array(age_list)
    age_max = np.max(age_list)
    age_min = np.min(age_list)
    age_mean = np.mean(age_list)
    age_std = np.std(age_list)

    meta = {}
    meta['age_max'] = age_max
    meta['age_min'] = age_min
    meta['age_mean'] = age_mean
    meta['age_std'] = age_std
    meta['max_token_id'] = max_token_id

    print("age_max:", age_max)
    print("age_min:", age_min)
    print("age_mean:", age_mean)
    print("age_std:", age_std)

    feature = np.vstack([ts for ts in feature])

    print(feature.shape)
    print(feature[:500, 16])

    feature_head = ['Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen',
                    'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total',
                    'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure',
                    'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']

    categorical = [1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for index, feature_name in enumerate(feature_head):
        if categorical[index]:
            continue
        print(f'{feature_name}:')
        now_feature = feature[:, index]
        mean = np.nanmean(now_feature)
        std = np.nanstd(now_feature)
        print(f"Mean:{mean}")
        print(f"Std:{std}")
        now_dict = {}
        now_dict['featrue'] = feature_name
        now_dict['mean'] = mean
        now_dict['std'] = std
        meta[str(index)] = now_dict

    with open(metadata_path, 'w') as json_file:
        json.dump(meta, json_file)


