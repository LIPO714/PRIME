import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from agumentation.d_agumentation import demogra_agumentation_bank
from agumentation.its_agumentation import its_agumentation_bank
from dataset.pretrainDataset_subpad import PretrainDataset, pretrain_collate_fn


def continuous_masking(note_data, max_token_id, min_token_id, n=5, l=0.03, min_len=100):
    '''
    :param note_data: B, 5 T
    :param note_attention_mask: B 5 T
    :param n: masking segment num
    :param l: masking segment length
    :param min_len: less than min_len not masking
    :return:
    '''
    B = len(note_data)
    device = note_data[0].device

    for i in range(B):
        note_data_mask = (note_data[i] != 1)
        note_len = torch.sum(note_data_mask, dim=1).cpu()  # 5
        L, T = note_data[i].shape
        masking = torch.full((L, T), False, dtype=torch.bool).to(device)
        mask_token = torch.randint(min_token_id, max_token_id + 1, size=(L, T), dtype=note_data[i].dtype).to(device)

        for j in range(L):
            now_len = note_len[j] - 2  # 去掉[cls]和[sep]
            if now_len <= min_len:
                continue
            # 计算段数
            if isinstance(n, float):
                now_n = int(n * now_len)
            else:
                now_n = n
            now_n = max(min(now_n, now_len // 2), 1)

            # 计算长度
            if isinstance(l, float):
                now_l = int(l * now_len)
            else:
                now_l = l
            now_l = max(now_l, 1)

            # mask
            for _ in range(now_n):
                t = np.random.randint(now_len - now_l + 1) + 1  # +1 是因为从[CLS]之后开始
                masking[j, t:t + now_l] = True

        note_data[i][masking] = mask_token[masking]

    # print("MASK:", masking[:2])


    # print("Before Mask:")
    # print("data:", note_data[:2, :1, :100])
    # print("note mask:", masking[:2, :1, :100])

    # print("After Mask:")
    # print("data:", note_data[:2, :1, :100])

    del masking, now_len, mask_token, note_data_mask, note_len
    return note_data

def reverse_order(note_data, bert, p=0.05):
    '''
    有一点小bug，如果连续2个以上的mask为truth，交换之后会少一个token。A B C --> B A C
    :param p: mask掉的概率
    :return:
    '''
    B = len(note_data)
    device = note_data[0].device

    for i in range(B):
        L, T = note_data[i].shape
        masking = torch.from_numpy(np.random.binomial(1, p, size=(L, T))).to(torch.bool).to(device)
        if bert == "BioBert":
            masking[note_data[i] == 102] = False
            masking[note_data[i] == 101] = False
            end_token_id = 102
        elif bert == "bioLongformer":
            masking[note_data[i] == 2] = False
            masking[note_data[i] == 0] = False
            end_token_id = 2
        masking[:, -1] = False

        sep_index = torch.nonzero(note_data[i] == end_token_id)
        j, k = sep_index[:, 0], sep_index[:, 1]
        masking[j, k-1] = False  # sep之前的一个token不能用于交换

        masking = masking.cpu()

        # print("Before Mask:")
        # print("note mask:", masking[:2, :1, :200])
        # print("data:", note_data[:2, :1, :200])

        note_data_copy = note_data[i].clone().to(device)

        masking_indices = torch.nonzero(masking)
        if masking_indices.numel() > 0:
            j, k = masking_indices[:, 0], masking_indices[:, 1]

            note_data[i][j, k] = note_data_copy[j, k+1]
            note_data[i][j, k+1] = note_data_copy[j, k]

        del j, k

    # print("After Mask:")
    # print("data:", note_data[:2, :1, :200])

    del note_data_copy, masking_indices, masking
    return note_data


def binomial_masking(note_data, bert, max_token_id, min_token_id, p=0.05):
    '''
    :param p: mask掉的概率
    :return:
    '''
    B = len(note_data)
    device = note_data[0].device
    for i in range(B):
        L, T = note_data[i].shape

        masking = torch.from_numpy(np.random.binomial(1, p, size=(L, T))).to(torch.bool).to(device)
        if bert == "BioBert":
            masking[note_data[i] == 102] = False
            masking[note_data[i] == 101] = False
            end_token_id = 102
        elif bert == "bioLongformer":
            masking[note_data[i] == 2] = False
            masking[note_data[i] == 0] = False
            end_token_id = 2

        mask_token = torch.randint(min_token_id, max_token_id + 1, size=(L, T), dtype=note_data[i].dtype).to(device)
        # print("Before Mask:")
        # print("data:", note_data[:2, :1, :100])
        # print("note mask:", masking[:2, :1, :100])
        note_data[i][masking] = mask_token[masking]
    # print("After Mask:")
    # print("data:", note_data[:2, :1, :100])
    del masking, mask_token
    return note_data

def note_agumentation_bank(note_data, bert, max_token_id, min_token_id=3):
    agumentation_choice = ['Continuous masking', 'Binomial masking', 'Reverse order']
    choice = random.choice(agumentation_choice)
    # choice = 'Binomial masking'
    # 1. 随机部分文段全部随机替换
    if choice == 'Continuous masking':
        note_data = continuous_masking(note_data, max_token_id, min_token_id)

    # 2. 随机反转token位置, 基于伯努利分布
    elif choice == 'Reverse order':
        note_data = reverse_order(note_data, bert)

    # 3. Binomial: 伯努利分布选择时刻替换
    elif choice == 'Binomial masking':
        note_data = binomial_masking(note_data, bert, max_token_id, min_token_id)

    return note_data


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_value = 68
    random.seed(seed_value)
    bert = "BioBert"  # "bioLongformer"

    dataset = PretrainDataset(data_path='../data/pretrain.pkl', device=device, note_length=256, note_num=5, bert=bert)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pretrain_collate_fn)

    meta_file = '../data/metadata.json'
    with open(meta_file, 'r') as json_file:
        meta = json.load(json_file)

    for index, batch in enumerate(dataloader):
        name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_tt, note_tau, note_mask, restore_index, restore_ts_label = batch
        print(index)
        print("ts_data.shape:", ts_data.shape)
        agu_choice = ['its', 'note']
        agu = random.choice(agu_choice)
        # agu = 'note'
        if agu == 'its':
            ts_data, ts_mask, ts_tau = its_agumentation_bank(ts_data, ts_tt, ts_mask, ts_tau, meta)
        else:  # 'note'
            if bert == "bioLongformer":
                max_token_id = meta['max_long_token_id']
                min_token_id = 3
            elif bert == "BioBert":
                max_token_id = meta['max_bio_token_id']
                min_token_id = 103
            note_data = note_agumentation_bank(note_data, bert, max_token_id, min_token_id)

        demogra = demogra_agumentation_bank(demogra, meta)

