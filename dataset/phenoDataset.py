import pickle
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from util import impute_ts


class Pheno24Dataset(Dataset):
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

        # # 1. 选取48小时的窗口【时间序列48小时；文本72小时, 最后时刻对齐】
        # ts_data, ts_mask, ts_tt, ts_tau, ts_len, start_time, end_time = self.cut_ts_window(48)
        # note_data, note_tt, note_tau = self.cut_note_window(end_time - 72, start_time, end_time)
        # 1. 读取数据
        ts_data = torch.tensor(self.now_data['ts_data']).float()
        ts_mask = torch.tensor(self.now_data['ts_mask']).float()
        ts_tt = torch.tensor(self.now_data['ts_tt']).float()
        ts_tau = torch.tensor(self.now_data['ts_tau']).float()
        note_data = self.now_data[self.note_index]
        note_tt = self.now_data['note_tt']
        note_tau = self.now_data['note_tau']
        label = torch.tensor(self.now_data['pheno_label']).float()

        # print("ts_tt:", ts_tt)
        # print("note_tt:", note_tt)

        # print("ts_data.shape:", ts_data.shape)
        # print("ts_mask.shape:", ts_mask.shape)
        # print("ts_tt.shape:", ts_tt.shape)
        # print("ts_tt:", ts_tt)
        # print("ts_tau.shape:", ts_tau.shape)
        #
        # print("note_data.shape:", len(note_data))
        # print("note_tt.shape:", note_tt.shape)
        # print("note_tt:", note_tt)
        # print("note_tau:", note_tau)

        # 2. 若时间序列长度大于ts_max_len，则截取一下最后的时刻
        if ts_data.shape[0] > self.ts_max_len:
            ts_data, ts_mask, ts_tt, ts_tau = self.cut_ts_data(ts_data, ts_mask, ts_tt, ts_tau)

        # 3. 选取最后note_num段note
        note_data, note_tt, note_tau, note_mask = self.choose_note(note_data, note_tt, note_tau)

        # print("---------2-----------")
        # print("note_data.shape:", len(note_data))
        # print("note_tt.shape:", note_tt.shape)
        # print("note_tt:", note_tt)
        # print("note_tau:", note_tau)
        # print("note_mask:", note_mask)

        # 3. 将note补0-->tensor
        note_data = pad_sequence(note_data, batch_first=True, padding_value=self.pad).to(self.device)
        # [5, maxlen] --> [maxlen, 5]
        note_data = note_data.transpose(1, 0)
        note_tt = torch.tensor(note_tt).float()
        note_tau = torch.tensor(note_tau).float()
        # print("---------3-----------")
        # print("note_data.shape:", note_data.shape)

        # 5. query tt
        query_tt = torch.arange(0, 25, 1).float()

        # 6. impute query ts data
        query_ts_data, query_ts_mask = self.impute_ts_data(query_tt, ts_data, ts_mask, ts_tt)

        return name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_tt, note_mask, note_tau, label, query_tt, query_ts_data, query_ts_mask, self.pad

    def cut_ts_data(self, ts_data, ts_mask, ts_tt, ts_tau):
        # 截取
        ts_data = ts_data[-self.ts_max_len:]
        ts_mask = ts_mask[-self.ts_max_len:]
        ts_tt = ts_tt[-self.ts_max_len:]
        ts_tau = ts_tau[-self.ts_max_len:]
        return ts_data, ts_mask, ts_tt, ts_tau

    def choose_note(self, note_data, note_tt, note_tau):
        # 选取note_num段最后；若不够note_num段，则需要补充
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
        # 1. 拆成离散型和连续型的
        continue_data = ts_data[:, :self.continue_dim]

        continue_data_mask = ts_mask[:, :self.continue_dim]

        # 2. 分别进行插值
        # continue date
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

        # cate data
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

            # 3. 合并
            query_ts_data = torch.cat([query_continue_data, query_cate_data], dim=1)
            query_ts_mask = torch.cat([query_continue_mask, query_cate_mask], dim=1)
        else:
            query_ts_data = query_continue_data
            query_ts_mask = query_continue_mask

        return query_ts_data, query_ts_mask


def pheno_collate_fn(batch):
    names, demogras, ts_datas, ts_tts, ts_masks, ts_taus, note_datas, note_tts, note_masks, note_taus, label, query_tt, query_ts_data, query_ts_mask, pads = zip(*batch)

    pad_list = list(pads)
    pad = pad_list[0]

    # 转换为列表
    names = list(names)  # list
    # note_datas = list(note_datas)
    # note_tts = list(note_tts)
    # note_masks = list(note_masks)
    # note_taus = list(note_taus)

    demogras = torch.stack(demogras)  # torch
    device = demogras.device
    ts_datas = pad_sequence(ts_datas, batch_first=True, padding_value=0).to(device)  # torch
    ts_tts = pad_sequence(ts_tts, batch_first=True, padding_value=0).to(device)  # torch
    ts_masks = pad_sequence(ts_masks, batch_first=True, padding_value=0).to(device)  # torch
    ts_taus = pad_sequence(ts_taus, batch_first=True, padding_value=0).to(device)  # torch

    note_datas = list(note_datas)
    note_datas = pad_sequence(note_datas, batch_first=True, padding_value=pad).to(device)  # torch
    # print("note_datas.shape:", note_datas.shape)
    note_datas = note_datas.transpose(2, 1)
    # print("note_datas.shape:", note_datas.shape)

    note_attention_mask = torch.zeros_like(note_datas)
    note_attention_mask[note_datas != pad] = 1
    note_attention_mask = note_attention_mask.to(device)
    # print("note_datas_attention_mask.shape:", note_datas_attention_mask)

    note_token_type = torch.zeros_like(note_datas).to(device)
    # print("note token type:", note_token_type.shape)

    note_tts = torch.stack(note_tts).to(device)
    note_taus = torch.stack(note_taus).to(device)
    note_masks = torch.stack(note_masks).to(device)
    label = torch.stack(label).to(device)

    query_tt = torch.stack(query_tt).to(device)
    query_ts_data = torch.stack(query_ts_data).to(device)
    query_ts_mask = torch.stack(query_ts_mask).to(device)

    # 1. note转成batch
    # 2. 对齐ts数据
    # 3. 转成tensor.to（device）

    return names, demogras, ts_datas, ts_tts, ts_masks, ts_taus, note_datas, note_attention_mask, note_token_type, note_tts, note_taus, note_masks, query_tt, query_ts_data, query_ts_mask, label


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_value = 68  # 可以替换为任何你想要的整数种子值
    random.seed(seed_value)

    # 创建CustomDataset实例
    dataset = Pheno24Dataset(data_path='../data/24pheno/train_24pheno_0.5.pkl', device=device, note_length=512, note_num=10, bert="bioLongformer")

    # 创建DataLoader实例，并使用自定义的collate_fn函数
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=pheno_collate_fn)

    # 使用DataLoader迭代数据
    for batch in dataloader:
        name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask, label = batch
        print("--------------------START---------------------------")
        # print("names:", name)
        # print("demogras:", demogra[:2])
        # print("ts_data.shape:", ts_data.shape)
        # print("ts_data:", ts_data[:2])
        # print("ts_tt.shape:", ts_tt.shape)
        # print("ts_tt:", ts_tt)
        # print("ts_tt.expand:", ts_tt.unsqueeze(2).expand(-1, -1, 17)[:2, :10])
        # print("ts_mask.shape:", ts_mask.shape)
        # print("ts_mask:", ts_mask[:2, :10])
        # print("ts_tau.shape:", ts_tau.shape)
        # print("ts_tau:", ts_tau[:2, :10])
        #
        # print("note_data:", note_data[:2, :2])
        # print("note_attention mask:", note_attention_mask[:2, :2])
        # print("note_token_type:", note_token_type[:2, :2])
        # print("note_tt.shape:", note_tt.shape)
        # print("note_tt:", note_tt[:2])
        # print("note_tau.shape:", note_tau.shape)
        # print("note_tau:", note_tau[:2])
        # print("note_mask.shape:", note_mask.shape)
        # print("note_mask:", note_mask[:2])

        print("label:", label.shape)
        print("label:", label)
