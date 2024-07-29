import pickle
import random

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from preprocess_pretrain_data import read_pkl, get_tau


AGE_MIN = 18.034033517250126
AGE_MAX = 89.08857461314054
AGE_MEAN = 62.140724966623495
AGE_STD = 16.960043591632854

def preprocess(data, cut_time, all_data, tokenizer, tokenizer_clinicbert):
    # dict_keys(['reg_ts', 'name', 'ts_tt', 'irg_ts', 'irg_ts_mask', 'text_data', 'text_time_to_start', 'ihm_label', 'ihm_task', 'los', 'decom_time', 'demographics', 'pheno_label'])
    print("len:", len(data))
    for i in tqdm(range(len(data)), desc="Processing", unit="iteration"):
        now_data = data[i]
        if now_data['ihm_task'] == 0:
            continue
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
            ts_before_num = np.sum(ts_tt < cut_time)
            ts_data = ts_data[:ts_before_num]
            ts_mask = ts_mask[:ts_before_num]
            ts_tt = ts_tt[:ts_before_num]
            ts_tau = ts_tau[:ts_before_num]

            note_before_num = np.sum(note_tt < cut_time)
            note_data = note_data[:note_before_num]
            note_tt = note_tt[:note_before_num]
            note_tau = note_tau[:note_before_num]

        ages = now_data['demographics'][0]
        if ages > 120:
            ages = AGE_MEAN
        now_data['demographics'][0] = (ages - AGE_MEAN) / AGE_STD

        # note-->token id
        note_encode_512_list = []
        note_encode_1024_list = []
        bio_cli_encode_256_list = []
        bio_cli_encode_512_list = []
        for index, note in enumerate(note_data):
            # if len(note.split(' ')) > 1024:
            #     maxer_512 += 1
            encode_text_512 = tokenizer.encode(note, max_length=512, truncation=True, return_tensors='pt')[0]
            encode_text_1024 = tokenizer.encode(note, max_length=1024, truncation=True, return_tensors='pt')[0]
            bio_cli_encode_text_256 = \
            tokenizer_clinicbert.encode(note, max_length=256, truncation=True, return_tensors='pt')[0]
            bio_cli_encode_text_512 = \
            tokenizer_clinicbert.encode(note, max_length=512, truncation=True, return_tensors='pt')[0]
            # print("512:", len(encode_text_512))
            # print("1024:", len(encode_text_1024))
            note_encode_512_list.append(encode_text_512)
            note_encode_1024_list.append(encode_text_1024)
            bio_cli_encode_256_list.append(bio_cli_encode_text_256)
            bio_cli_encode_512_list.append(bio_cli_encode_text_512)

        patient = {}
        patient['name'] = now_data['name']
        patient['demographics'] = now_data['demographics']
        patient['48ihm_label'] = now_data['ihm_label']
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
    return all_data


if __name__ == '__main__':
    train_data_path = '../data/train_ts_note_data.pkl'
    test_data_path = '../data/test_ts_note_data.pkl'
    cut_time = '48'
    percent = 0.01
    train_val_percent = 0.8  # 0.8
    out_train_data_path = '../data/48ihm/train_48ihm_'+str(percent)+'.pkl'
    out_val_data_path = '../data/48ihm/val_48ihm_'+str(percent)+'.pkl'
    out_test_data_path = '../data/48ihm/test_48ihm_'+str(percent)+'.pkl'

    train_data = read_pkl(train_data_path)
    test_data = read_pkl(test_data_path)

    tokenizer = AutoTokenizer.from_pretrained('../Clinical-Longformer')
    tokenizer_clinicbert = AutoTokenizer.from_pretrained("../Bio_ClinicalBERT")

    train_list = []
    train_list = preprocess(train_data, cut_time, train_list, tokenizer, tokenizer_clinicbert)
    test_list = []
    test_list = preprocess(test_data, cut_time, test_list, tokenizer, tokenizer_clinicbert)
    print("ORIGIN NUM:")
    train_list_num = len(train_list)
    test_list_num = len(test_list)
    print("train num:", train_list_num)
    print("test num:", test_list_num)

    random.shuffle(train_list)
    random.shuffle(test_list)

    train_list = train_list[:int(train_list_num*percent)]
    print("After percent NUM:")
    train_list_num = len(train_list)
    print("train num:", train_list_num)
    print("test num:", test_list_num)

    train_num = int(train_list_num * train_val_percent)
    val_num = train_list_num - train_num
    train_data = train_list[:train_num]
    val_data = train_list[train_num:]
    print("dataset NUM:")
    print("train num:", train_num)
    print("val num:", val_num)
    print("test num:", test_list_num)


    with open(out_train_data_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(out_val_data_path, 'wb') as f:
        pickle.dump(val_data, f)
    with open(out_test_data_path, 'wb') as f:
        pickle.dump(test_list, f)
