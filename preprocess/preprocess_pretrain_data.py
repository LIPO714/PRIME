'''
merge train and test data for pretrain
preprocess pretrain data
'''
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer

AGE_MIN = 18.034033517250126
AGE_MAX = 89.08857461314054
AGE_MEAN = 62.140724966623495
AGE_STD = 16.960043591632854

def read_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_tau(ts_tt, ts_mask):
    # L, [L, K]
    tmp_time = ts_mask * np.expand_dims(ts_tt, axis=-1)  # [L,K]

    new_mask = ts_mask.copy()
    new_mask[0, :] = 1
    tmp_time[new_mask == 0] = np.nan

    # padding the missing value with the next value
    df1 = pd.DataFrame(tmp_time)
    df1 = df1.fillna(method='ffill')
    tmp_time = np.array(df1)

    tmp_time[1:, :] -= tmp_time[:-1, :]
    del new_mask
    return tmp_time * ts_mask


def preprocess(data, cut_time, all_data, age_list, tokenizer, tokenizer_clinicbert):
    # dict_keys(['reg_ts', 'name', 'ts_tt', 'irg_ts', 'irg_ts_mask', 'text_data', 'text_time_to_start', 'ihm_label', 'ihm_task', 'los', 'decom_time', 'demographics', 'pheno_label'])
    maxer_512 = 0
    for i in tqdm(range(len(data)), desc="Processing", unit="iteration"):
        now_data = data[i]
        ts_data = now_data['irg_ts']
        ts_mask = now_data['irg_ts_mask']
        ts_tt = np.array(now_data['ts_tt'])
        note_data = now_data['text_data']
        note_tt = np.array(now_data['text_time_to_start'][0])

        start_time = ts_tt[0]
        ts_tt = ts_tt - start_time
        note_tt = note_tt - start_time

        ts_tau = get_tau(ts_tt, ts_mask)

        # print("ts_tt:", ts_tt)
        # print("ts_mask:", ts_mask)
        # print("ts_tau:", ts_tau)

        note_tau = note_tt[1:] - note_tt[:-1]
        note_tau = np.concatenate(([note_tt[0]], note_tau))

        if cut_time is not None:
            cut_time = float(cut_time)
            ts_before_num = np.sum(ts_tt <= cut_time + 1e-6)
            ts_data = ts_data[:ts_before_num]
            ts_mask = ts_mask[:ts_before_num]
            ts_tt = ts_tt[:ts_before_num]
            ts_tau = ts_tau[:ts_before_num]

            note_before_num = np.sum(note_tt <= cut_time + 1e-6)
            note_data = note_data[:note_before_num]
            note_tt = note_tt[:note_before_num]
            note_tau = note_tau[:note_before_num]

        if len(ts_tt) < 8:
            continue

        if cut_time is not None:
            if len(note_data) < 3:
                continue
        else:
            density = ts_tt[-1] / len(note_data)
            if density > 16:
                continue

        ages = now_data['demographics'][0]
        if ages < 120:
            age_list.append(ages)
        else:
            ages = AGE_MEAN
        now_data['demographics'][0] = (ages - AGE_MEAN) / AGE_STD

        # note-->token id
        note_encode_512_list = []
        note_encode_1024_list = []
        bio_cli_encode_256_list = []
        bio_cli_encode_512_list = []

        for index, note in enumerate(note_data):
            # if len(note.split(' ')) > 512:
            #     maxer_512 += 1
            encode_text_512 = tokenizer.encode(note, max_length=512, truncation=True, return_tensors='pt')[0]
            encode_text_1024 = tokenizer.encode(note, max_length=1024, truncation=True, return_tensors='pt')[0]
            bio_cli_encode_text_256 = tokenizer_clinicbert.encode(note, max_length=256, truncation=True, return_tensors='pt')[0]
            bio_cli_encode_text_512 = tokenizer_clinicbert.encode(note, max_length=512, truncation=True, return_tensors='pt')[0]

            note_encode_512_list.append(encode_text_512)
            note_encode_1024_list.append(encode_text_1024)
            bio_cli_encode_256_list.append(bio_cli_encode_text_256)
            bio_cli_encode_512_list.append(bio_cli_encode_text_512)


        patient = {}
        patient['name'] = now_data['name']
        print(patient['name'])
        patient['demographics'] = now_data['demographics']
        patient['ts_data'] = ts_data
        patient['ts_mask'] = ts_mask
        patient['ts_tt'] = ts_tt
        patient['ts_tau'] = ts_tau
        patient['note_data'] = note_data
        patient['note_encode_512_data'] = note_encode_512_list
        patient['note_encode_1024_data'] = note_encode_1024_list
        patient['note_encode_bio_256_data'] = bio_cli_encode_256_list
        patient['note_encode_bio_512_data'] = bio_cli_encode_512_list
        patient['note_tt'] = note_tt
        patient['note_tau'] = note_tau
        all_data.append(patient)
    print("Num:", len(all_data))
    # print("Maxer 512:", maxer_512)
    return all_data, age_list


if __name__ == '__main__':
    train_data_path = '../data/train_ts_note_data.pkl'
    # test_data_path = '../data/test_ts_note_data.pkl'
    cut_time = '48'  # None  # '48'
    if cut_time is not None:
        out_data_path = './data/pretrain_' + cut_time + '.pkl'
    else:
        out_data_path = '../data/pretrain.pkl'

    train_data = read_pkl(train_data_path)
    # test_data = read_pkl(test_data_path)

    tokenizer = AutoTokenizer.from_pretrained('../Clinical-Longformer')  # padding用的是1
    tokenizer_clinicbert = AutoTokenizer.from_pretrained("../Bio_ClinicalBERT")
    max_token_id = max(tokenizer.get_vocab().values())
    print("max:", max_token_id)
    max_token_id = max(tokenizer_clinicbert.get_vocab().values())
    print("max_token_bio:", max_token_id)
    # cls_token_id = tokenizer.cls_token_id
    # sep_token_id = tokenizer.sep_token_id
    #
    # print("[CLS] Token ID:", cls_token_id)
    # print("[SEP] Token ID:", sep_token_id)
    # text = ["My name is:", "My age:" , "I"]
    # encode_text = tokenizer.batch_encode_plus(text, padding=True, truncation=True, return_tensors='pt')
    # print(encode_text)

    all_data = []
    age_list = []

    all_data, age_list = preprocess(train_data, cut_time, all_data, age_list, tokenizer, tokenizer_clinicbert)
    # all_data, age_list = preprocess(test_data, cut_time, all_data, age_list, tokenizer, tokenizer_clinicbert)

    age_list = np.array(age_list)
    # age_min = np.min(age_list)
    # age_max = np.max(age_list)
    # age_mean = np.mean(age_list)
    # age_std = np.std(age_list)
    # print("age_min:", age_min)
    # print("age_max:", age_max)
    # print("age_mean:", age_mean)
    # print("age_std:", age_std)

    with open(out_data_path, 'wb') as f:
        pickle.dump(all_data, f)
