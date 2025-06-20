import pickle
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from util import impute_ts


class Ihm48Dataset(Dataset):
    def __init__(self, args, data_path, device, note_length, note_num=10, bert="bioLongformer", ts_max_len=200):
        self.data = self.read_pkl(data_path)
        self.device = device
        self.note_length = str(note_length)
        self.note_num = note_num
        self.bert = bert
        self.ts_max_len = ts_max_len
        self.conti_impute_type = args.conti_impute_type
        self.cate_impute_type = args.cate_impute_type
        if bert == 'bioLongformer':
            self.note_index = 'note_encode_' + self.note_length + '_data'
            self.empty = torch.tensor([0, 2])
            self.pad = 1
        elif bert == 'BioBert':
            self.note_index = 'note_encode_bio_' + self.note_length + '_data'
            self.empty = torch.tensor([101, 102])
            self.pad = 0

        # categorical
        self.cate_dim = args.cate_dim
        self.continue_dim = args.var_dim - self.cate_dim
        cate_type_str = args.cate_type
        cate_type_list = cate_type_str.split("_")
        self.cate_type = []
        for i in cate_type_list:
            self.cate_type.append(int(i))

    def read_pkl(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.now_data = self.data[index]
        name = self.now_data['name']
        demogra = self.now_data['demographics']
        demogra = torch.tensor(demogra).to(self.device)

        ts_data = torch.tensor(self.now_data['ts_data']).float()
        ts_mask = torch.tensor(self.now_data['ts_mask']).float()
        ts_tt = torch.tensor(self.now_data['ts_tt']).float()
        ts_tau = torch.tensor(self.now_data['ts_tau']).float()
        note_data = self.now_data[self.note_index]
        note_tt = self.now_data['note_tt']
        note_tau = self.now_data['note_tau']
        label = torch.tensor(self.now_data['48ihm_label']).float()

        if ts_data.shape[0] > self.ts_max_len:
            ts_data, ts_mask, ts_tt, ts_tau = self.cut_ts_data(ts_data, ts_mask, ts_tt, ts_tau)

        note_data, note_tt, note_tau, note_mask = self.choose_note(note_data, note_tt, note_tau)

        note_data = pad_sequence(note_data, batch_first=True, padding_value=self.pad).to(self.device)
        # [5, maxlen] --> [maxlen, 5]
        note_data = note_data.transpose(1, 0)
        note_tt = torch.tensor(note_tt).float()
        note_tau = torch.tensor(note_tau).float()

        # 5. query tt
        query_tt = torch.arange(0, 49, 1).float()

        # 6. impute query ts data
        query_ts_data, query_ts_mask = self.impute_ts_data(query_tt, ts_data, ts_mask, ts_tt)

        return name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_tt, note_mask, note_tau, label, query_tt, query_ts_data, query_ts_mask, self.pad

    def cut_ts_data(self, ts_data, ts_mask, ts_tt, ts_tau):
        ts_data = ts_data[-self.ts_max_len:]
        ts_mask = ts_mask[-self.ts_max_len:]
        ts_tt = ts_tt[-self.ts_max_len:]
        ts_tau = ts_tau[-self.ts_max_len:]
        return ts_data, ts_mask, ts_tt, ts_tau

    def choose_note(self, note_data, note_tt, note_tau):
        length = self.note_num
        note_len = len(note_data)
        note_mask = torch.ones(length)
        if note_len < length:
            padding_num = length - note_len
            empty_note = [self.empty for _ in range(padding_num)]
            note_data = note_data + empty_note
            note_tt = np.concatenate((note_tt, [0]*padding_num))
            note_tau = np.concatenate((note_tau, [0]*padding_num))
            note_mask[-padding_num:] = 0
        elif note_len > length:
            note_data = note_data[-length:]
            note_tt = note_tt[-length:]
            note_tau = note_tau[-length:]

        return note_data, note_tt, note_tau, note_mask

    def impute_ts_data(self, query_ts_tt, ts_data, ts_mask, ts_tt):
        continue_data = ts_data[:, :self.continue_dim]

        continue_data_mask = ts_mask[:, :self.continue_dim]

        if self.conti_impute_type == "linear":
            query_continue_data_plus, query_ts_dt_plus, query_continue_mask_plus = impute_ts(query_ts_tt, continue_data,
                                                                                             continue_data_mask, ts_tt,
                                                                                             sort="+")
            query_continue_data_minus, query_ts_dt_minus, query_continue_mask_minus = impute_ts(query_ts_tt,
                                                                                                continue_data,
                                                                                                continue_data_mask,
                                                                                                ts_tt, sort="-")

            sum = query_ts_dt_minus + query_ts_dt_plus
            sum[sum == 0] = 1

            query_continue_data = query_continue_data_plus + (
                        query_continue_data_minus - query_continue_data_plus) * query_ts_dt_plus / (sum)
            query_continue_mask = query_continue_mask_plus

        elif self.conti_impute_type == "backward":
            query_continue_data, _, query_continue_mask = impute_ts(query_ts_tt, continue_data, continue_data_mask,
                                                                    ts_tt, sort="+")
        elif self.conti_impute_type == "forward":
            query_continue_data, _, query_continue_mask = impute_ts(query_ts_tt, continue_data, continue_data_mask,
                                                                    ts_tt, sort="-")

        if self.cate_dim > 0:
            cate_data = ts_data[:, self.continue_dim:]
            cate_data_mask = ts_mask[:, self.continue_dim:]
            if self.cate_impute_type == "linear":
                query_cate_data_plus, query_ts_dt_plus, query_cate_mask_plus = impute_ts(query_ts_tt, cate_data,
                                                                                         cate_data_mask, ts_tt,
                                                                                         sort="+")
                query_cate_data_minus, query_ts_dt_minus, query_cate_mask_minus = impute_ts(query_ts_tt, cate_data,
                                                                                            cate_data_mask, ts_tt,
                                                                                            sort="-")

                mask = query_ts_dt_plus > query_ts_dt_minus

                query_cate_data = query_cate_data_plus
                query_cate_data[mask] = query_cate_data_minus[mask]
                query_cate_mask = query_cate_mask_plus
                query_cate_mask[query_cate_mask_minus > query_cate_mask_plus] = query_cate_data_minus[
                    query_cate_mask_minus > query_cate_mask_plus]

            elif self.cate_impute_type == "backward":
                query_cate_data, _, query_cate_mask = impute_ts(query_ts_tt, cate_data, cate_data_mask, ts_tt, sort="+")
            elif self.cate_impute_type == "forward":
                query_cate_data, _, query_cate_mask = impute_ts(query_ts_tt, cate_data, cate_data_mask, ts_tt, sort="-")

            query_ts_data = torch.cat([query_continue_data, query_cate_data], dim=1)
            query_ts_mask = torch.cat([query_continue_mask, query_cate_mask], dim=1)
        else:
            query_ts_data = query_continue_data
            query_ts_mask = query_continue_mask

        return query_ts_data, query_ts_mask


def ihm_collate_fn(batch):
    names, demogras, ts_datas, ts_tts, ts_masks, ts_taus, note_datas, note_tts, note_masks, note_taus, label, query_tts, query_ts_data, query_ts_mask, pads = zip(*batch)

    pad_list = list(pads)
    pad = pad_list[0]

    names = list(names)

    demogras = torch.stack(demogras)  # torch
    device = demogras.device
    ts_datas = pad_sequence(ts_datas, batch_first=True, padding_value=0).to(device)  # torch
    ts_tts = pad_sequence(ts_tts, batch_first=True, padding_value=0).to(device)  # torch
    ts_masks = pad_sequence(ts_masks, batch_first=True, padding_value=0).to(device)  # torch
    ts_taus = pad_sequence(ts_taus, batch_first=True, padding_value=0).to(device)  # torch

    note_datas = list(note_datas)
    note_datas = pad_sequence(note_datas, batch_first=True, padding_value=pad).to(device)  # torch
    note_datas = note_datas.transpose(2, 1)

    note_attention_mask = torch.zeros_like(note_datas).to(device)
    note_attention_mask[note_datas != pad] = 1
    note_attention_mask = note_attention_mask.to(device)

    note_token_type = torch.zeros_like(note_datas).to(device)

    note_tts = torch.stack(note_tts).to(device)
    note_taus = torch.stack(note_taus).to(device)
    note_masks = torch.stack(note_masks).to(device)
    label = torch.stack(label).to(device)

    query_tts = torch.stack(query_tts).to(device)
    query_ts_data = torch.stack(query_ts_data).to(device)
    query_ts_mask = torch.stack(query_ts_mask).to(device)

    return names, demogras, ts_datas, ts_tts, ts_masks, ts_taus, note_datas, note_attention_mask, note_token_type, note_tts, note_taus, note_masks, query_tts, query_ts_data, query_ts_mask, label


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_value = 68
    random.seed(seed_value)

    dataset = Ihm48Dataset(data_path='../data/48ihm/train_48ihm_0.5.pkl', device=device, note_length=512, note_num=10, bert="bioLongformer")

    print("len:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=ihm_collate_fn)

    sum = 0
    for batch in dataloader:
        name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask, label = batch
        print("--------------------START---------------------------")
        print("ts_data:", ts_data.shape)
        if ts_data.shape[1] > 200:
            sum += 1

        print("label:", label.shape)
        print("label:", label)

    print("sum:", sum)