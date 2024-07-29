import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from agumentation.d_agumentation import demogra_agumentation_bank
from agumentation.its_agumentation import its_agumentation_bank
from dataset.pretrainDataset import PretrainDataset, pretrain_collate_fn


def continuous_masking(note_data, max_token_id, min_token_id, n=5, l=0.03, min_len=100):
    '''
    :param note_data: B 5 T
    :param note_attention_mask: B 5 T
    :param n: masking segment num
    :param l: masking segment length
    :param min_len: less than min_len not masking
    :return:
    '''
    B, L, T = note_data.shape
    note_data_mask = (note_data != 1)
    note_len = torch.sum(note_data_mask, dim=2).cpu()  # B 5

    masking = torch.full((B, L, T), False, dtype=torch.bool).to(ts_data.device)
    for i in range(B):
        for j in range(L):
            now_len = note_len[i, j] - 2  # 去掉[cls]和[sep]
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
                masking[i, j, t:t + now_l] = True

    # print("MASK:", masking[:2])

    mask_token = torch.randint(min_token_id, max_token_id+1, size=(B, L, T), dtype=note_data.dtype).to(note_data.device)
    # print("Before Mask:")
    # print("data:", note_data[:2, :1, :100])
    # print("note mask:", masking[:2, :1, :100])
    note_data[masking] = mask_token[masking]
    # print("After Mask:")
    # print("data:", note_data[:2, :1, :100])

    del masking, now_len, mask_token, note_data_mask, note_len
    return note_data

def reverse_order(note_data, p=0.05):
    '''
    有一点小bug，如果连续2个以上的mask为truth，交换之后会少一个token。A B C --> B A C
    :param p: mask掉的概率
    :return:
    '''
    B, L, T = note_data.shape

    masking = torch.from_numpy(np.random.binomial(1, p, size=(B, L, T))).to(torch.bool).to(note_data.device)
    masking[note_data == 2] = False
    masking[note_data == 0] = False
    masking[:, :, -1] = False
    sep_index = torch.nonzero(note_data == 2)
    i, j, k = sep_index[:, 0], sep_index[:, 1], sep_index[:, 2]
    masking[i, j, k-1] = False  # sep之前的一个token不能用于交换 一会要改

    masking = masking.cpu()

    # print("Before Mask:")
    # print("note mask:", masking[:2, :1, :200])
    # print("data:", note_data[:2, :1, :200])

    note_data_copy = note_data.clone().to(note_data.device)

    masking_indices = torch.nonzero(masking)
    if masking_indices.numel() > 0:
        i, j, k = masking_indices[:, 0], masking_indices[:, 1], masking_indices[:, 2]

        note_data[i, j, k] = note_data_copy[i, j, k+1]
        note_data[i, j, k+1] = note_data_copy[i, j, k]

    del i, j, k

    # print("After Mask:")
    # print("data:", note_data[:2, :1, :200])

    del note_data_copy, masking_indices, masking
    return note_data


def binomial_masking(note_data, max_token_id, min_token_id, p=0.05):
    '''
    :param p: mask掉的概率
    :return:
    '''
    B, L, T = note_data.shape
    masking = torch.from_numpy(np.random.binomial(1, p, size=(B, L, T))).to(torch.bool).to(note_data.device)
    masking[note_data == 2] = False
    masking[note_data == 0] = False

    mask_token = torch.randint(min_token_id, max_token_id + 1, size=(B, L, T), dtype=note_data.dtype).to(
        note_data.device)
    # print("Before Mask:")
    # print("data:", note_data[:2, :1, :100])
    # print("note mask:", masking[:2, :1, :100])
    note_data[masking] = mask_token[masking]
    # print("After Mask:")
    # print("data:", note_data[:2, :1, :100])
    del masking, mask_token
    return note_data

def note_agumentation_bank(note_data, max_token_id, min_token_id=3):
    agumentation_choice = ['Continuous masking', 'Binomial masking', 'Reverse order']
    choice = random.choice(agumentation_choice)
    # choice = 'Reverse order'
    # 1. 随机部分文段全部随机替换
    if choice == 'Continuous masking':
        note_data = continuous_masking(note_data, max_token_id, min_token_id)

    # 2. 随机反转token位置, 基于伯努利分布
    elif choice == 'Reverse order':
        note_data = reverse_order(note_data)

    # 3. Binomial: 伯努利分布选择时刻替换
    elif choice == 'Binomial masking':
        note_data = binomial_masking(note_data, max_token_id, min_token_id)

    return note_data


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_value = 68
    random.seed(seed_value)

    dataset = PretrainDataset(data_path='../data/pretrain.pkl', device=device, note_length=1024)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pretrain_collate_fn)

    meta_file = '../data/metadata.json'
    with open(meta_file, 'r') as json_file:
        meta = json.load(json_file)

    for index, batch in enumerate(dataloader):
        name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask = batch
        print(index)
        agu_choice = ['its', 'note']
        agu = random.choice(agu_choice)
        if agu == 'its':
            ts_data, ts_mask, ts_tau = its_agumentation_bank(ts_data, ts_tt, ts_mask, ts_tau, meta)
        else:  # 'note'
            note_data = note_agumentation_bank(note_data, meta['max_token_id'])

        demogra = demogra_agumentation_bank(demogra, meta)

