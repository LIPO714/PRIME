from preprocess.preprocess_pretrain_data import read_pkl
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm



label = []
for i in range(0, 17):
    label.append(2 ** i)
label = np.array(label)
print(label)

def preprocess(data, cut_time, event_dict):
    for i in tqdm(range(len(data)), desc="Processing", unit="iteration"):
        self_event_dict = {}
        now_data = data[i]
        if now_data['name'] != "63_episode1_timeseries.csv":
            continue
        ts_mask = now_data['irg_ts_mask']
        ts_tt = np.array(now_data['ts_tt'])
        ts_data = now_data['irg_ts']
        start_time = ts_tt[0]
        ts_tt = ts_tt - start_time  # 开始时间对齐到0

        if cut_time is not None:
            cut_time = float(cut_time)
            ts_before_num = np.sum(ts_tt <= cut_time + 1e-6)
            ts_mask = ts_mask[:ts_before_num]
            ts_data = ts_data[:ts_before_num]
            print(ts_data[:, 3:7])

        event = np.sum(ts_mask * label, axis=1)
        length = event.shape[0]
        for i in range(length):
            if event[i] in event_dict.keys():
                event_dict[event[i]] += 1
            else:
                event_dict[event[i]] = 1

            if event[i] in self_event_dict.keys():
                self_event_dict[event[i]] += 1
            else:
                self_event_dict[event[i]] = 1

        # print(self_event_dict)

    return event_dict


if __name__ == '__main__':
    train_data_path = './data/train_ts_note_data.pkl'
    test_data_path = './data/test_ts_note_data.pkl'
    cut_time = '48'  # None  # '48'
    if cut_time is not None:
        out_data_path = './data/pretrain_' + cut_time + '.pkl'
    else:
        out_data_path = '../data/pretrain.pkl'

    train_data = read_pkl(train_data_path)
    test_data = read_pkl(test_data_path)

    event_dict = {}

    event_dict = preprocess(train_data, cut_time, event_dict)
    event_dict = preprocess(test_data, cut_time, event_dict)

    more = 0
    less = 0
    for key in event_dict.keys():
        print(f'{key}:{event_dict[key]}')
        if event_dict[key] >= 5:
            more += 1
        else:
            less += 1

    print("more:", more)
    print("less:", less)

