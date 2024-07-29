import pickle
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from util import impute_ts
from tqdm import tqdm

AGE_MEAN = 62.140724966623495
AGE_STD = 16.960043591632854


def analysis_24pheno_data():
    dataset = "0.01"
    path_list = ['../data/24pheno/train_24pheno_', '../data/24pheno/val_24pheno_', '../data/24pheno/test_24pheno_']

    age_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    gender_list = [0, 0]

    ts_len_list = []
    ts_valid = []
    ts_invalid = []
    ts_tt_list = []
    note_num_list = []
    dead_num = 0
    num = 0

    for path in path_list:
        with open(path+dataset+'.pkl', 'rb') as file:
            data = pickle.load(file)

        note_index = 'note_encode_512_data'

        for now_data in tqdm(data):
            num += 1
            name = now_data['name']
            demogra = now_data['demographics']
            ages = demogra[0] * AGE_STD + AGE_MEAN
            gender = demogra[1]

            age_list[int(ages / 10)] += 1
            gender_list[int(gender)] += 1

            ts_data = torch.tensor(now_data['ts_data']).float()
            ts_mask = torch.tensor(now_data['ts_mask']).float()
            ts_tt = torch.tensor(now_data['ts_tt']).float()
            ts_tau = torch.tensor(now_data['ts_tau']).float()
            note_data = now_data[note_index]
            note_tt = now_data['note_tt']
            note_tau = now_data['note_tau']

            label = torch.tensor(now_data['pheno_label']).float()

            ts_len_list.append(ts_data.shape[0])
            ts_valid.append(torch.sum(ts_mask))
            ts_invalid.append(ts_data.shape[0]*ts_data.shape[1] - torch.sum(ts_mask))

            ts_tt_list.append(ts_tt[-1])

            note_num_list.append(len(note_data))


    print("24PHENO:"+dataset)

    print("num:", num)
    print("age:", age_list)
    print("gender:", gender_list)

    ts_len_list = np.array(ts_len_list)
    ts_valid = np.array(ts_valid)
    ts_invalid = np.array(ts_invalid)
    ts_tt_list = np.array(ts_tt_list)
    note_num_list = np.array(note_num_list)

    print("ts mean len:", np.mean(ts_len_list))
    ts_valid_sum = np.sum(ts_valid)
    ts_invalid_sum = np.sum(ts_invalid)
    print("ts_valid/all:", ts_valid_sum/(ts_valid_sum + ts_invalid_sum))
    print("ts tt mean:", np.mean(ts_tt_list))
    ts_tt_less_than_24 = ts_tt_list <= 24
    ts_tt_less_than_48 = ts_tt_list <= 48
    print("ts tt less than 24:", np.sum(ts_tt_less_than_24))
    print("ts tt less than 48:", np.sum(ts_tt_less_than_48))
    print("note num mean:", np.mean(note_num_list))


def analysis_48ihm_data():
    dataset = "0.01"
    path_list = ['../data/48ihm/train_48ihm_', '../data/48ihm/val_48ihm_', '../data/48ihm/test_48ihm_']

    age_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    gender_list = [0, 0]

    ts_len_list = []
    ts_valid = []
    ts_invalid = []
    ts_tt_list = []
    note_num_list = []
    dead_num = 0
    num = 0

    for path in path_list:
        with open(path+dataset+'.pkl', 'rb') as file:
            data = pickle.load(file)

        note_index = 'note_encode_512_data'

        for now_data in tqdm(data):
            num += 1
            name = now_data['name']
            demogra = now_data['demographics']
            ages = demogra[0] * AGE_STD + AGE_MEAN
            gender = demogra[1]

            age_list[int(ages / 10)] += 1
            gender_list[int(gender)] += 1

            ts_data = torch.tensor(now_data['ts_data']).float()
            ts_mask = torch.tensor(now_data['ts_mask']).float()
            ts_tt = torch.tensor(now_data['ts_tt']).float()
            ts_tau = torch.tensor(now_data['ts_tau']).float()
            note_data = now_data[note_index]
            note_tt = now_data['note_tt']
            note_tau = now_data['note_tau']

            label = torch.tensor(now_data['48ihm_label']).float()

            ts_len_list.append(ts_data.shape[0])
            ts_valid.append(torch.sum(ts_mask))
            ts_invalid.append(ts_data.shape[0]*ts_data.shape[1] - torch.sum(ts_mask))

            ts_tt_list.append(ts_tt[-1])

            note_num_list.append(len(note_data))

            if label != 0:
                dead_num += 1

    print("48IHM:"+dataset)

    print("num:", num)
    print("age:", age_list)
    print("gender:", gender_list)

    ts_len_list = np.array(ts_len_list)
    ts_valid = np.array(ts_valid)
    ts_invalid = np.array(ts_invalid)
    ts_tt_list = np.array(ts_tt_list)
    note_num_list = np.array(note_num_list)

    print("ts mean len:", np.mean(ts_len_list))
    ts_valid_sum = np.sum(ts_valid)
    ts_invalid_sum = np.sum(ts_invalid)
    print("ts_valid/all:", ts_valid_sum/(ts_valid_sum + ts_invalid_sum))
    print("ts tt mean:", np.mean(ts_tt_list))
    ts_tt_less_than_24 = ts_tt_list <= 24
    ts_tt_less_than_48 = ts_tt_list <= 48
    print("ts tt less than 24:", np.sum(ts_tt_less_than_24))
    print("ts tt less than 48:", np.sum(ts_tt_less_than_48))
    print("note num mean:", np.mean(note_num_list))
    print("dead num:", dead_num)


def analysis_pretrain_data():
    path = '../data/pretrain.pkl'
    with open(path, 'rb') as file:
        data = pickle.load(file)

    note_index = 'note_encode_512_data'

    age_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    gender_list = [0, 0]

    ts_len_list = []
    ts_valid = []
    ts_invalid = []
    ts_tt_list = []
    note_num_list = []
    num = 0

    for now_data in tqdm(data):
        num += 1
        name = now_data['name']
        demogra = now_data['demographics']
        ages = demogra[0] * AGE_STD + AGE_MEAN
        gender = demogra[1]

        age_list[int(ages / 10)] += 1
        gender_list[int(gender)] += 1

        ts_data = torch.tensor(now_data['ts_data']).float()
        ts_mask = torch.tensor(now_data['ts_mask']).float()
        ts_tt = torch.tensor(now_data['ts_tt']).float()
        ts_tau = torch.tensor(now_data['ts_tau']).float()
        note_data = now_data[note_index]
        note_tt = now_data['note_tt']
        note_tau = now_data['note_tau']

        ts_len_list.append(ts_data.shape[0])
        ts_valid.append(torch.sum(ts_mask))
        ts_invalid.append(ts_data.shape[0]*ts_data.shape[1] - torch.sum(ts_mask))
        ts_tt_list.append(ts_tt[-1])

        note_num_list.append(len(note_data))

    print("age:", age_list)
    print("gender:", gender_list)

    ts_len_list = np.array(ts_len_list)
    ts_valid = np.array(ts_valid)
    ts_invalid = np.array(ts_invalid)
    ts_tt_list = np.array(ts_tt_list)
    note_num_list = np.array(note_num_list)

    print("ts mean len:", np.mean(ts_len_list))
    ts_valid_sum = np.sum(ts_valid)
    ts_invalid_sum = np.sum(ts_invalid)
    print("ts_valid/all:", ts_valid_sum/(ts_valid_sum + ts_invalid_sum))
    print("ts tt mean:", np.mean(ts_tt_list))
    ts_tt_less_than_24 = ts_tt_list <= 24
    ts_tt_less_than_48 = ts_tt_list <= 48
    print("ts tt less than 24:", np.sum(ts_tt_less_than_24))
    print("ts tt less than 48:", np.sum(ts_tt_less_than_48))
    print("note num mean:", np.mean(note_num_list))


if __name__ == '__main__':
    # analysis_pretrain_data()
    # analysis_48ihm_data()
    analysis_24pheno_data()