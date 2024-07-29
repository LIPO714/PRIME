import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.pretrainDataset import PretrainDataset, pretrain_collate_fn


def noise_addition(ts_data, ts_mask, meta):
    add_channel_choice = ['1', '2', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    add_num = 6  # 加噪通道数
    noise_percent = 0.01  # 另一个模型说是10%，但是好像幅度太大了，我觉得1%就足够了
    add_channel = random.sample(add_channel_choice, add_num)

    # print("add channel:", add_channel)
    # print("ts before:", ts_data[:2, :5])

    b, l, k = ts_data.shape
    noise = torch.zeros_like(ts_data).float().to(ts_data.device)
    for channel in add_channel:
        channel_noise = (torch.randn(b, l) * meta[channel]['std']*noise_percent).to(ts_data.device)
        noise[:, :, int(channel)] = channel_noise
    ts_data = ts_data + noise
    ts_data[ts_mask==0] = 0

    # print("ts after:", ts_data[:2, :5])

    del noise, channel_noise
    return ts_data

def continuous_masking(ts_data, ts_tt, ts_mask, ts_tau, n=2, l=0.05, min_len=10):
    '''
    :param ts_data: B L K
    :param ts_tt: B L
    :param ts_mask: B L K
    :param n: masking segment num
    :param l: masking segment length
    :param min_len: less than min_len not masking
    :return:
    '''
    B, L, K = ts_data.shape
    ts_tt_mask = ts_tt > 0
    ts_tt_mask[:, 0] = 1
    ts_len = torch.sum(ts_tt_mask, dim=1).cpu()

    masking = torch.full((B, L), False, dtype=torch.bool).to(ts_data.device)
    for i in range(B):
        now_len = ts_len[i]
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
            t = np.random.randint(now_len - now_l + 1)
            masking[i, t:t + now_l] = True

    # print("MASK:", masking[:2])

    masking = masking.unsqueeze(2).expand(B, L, K)
    # print("Before Mask:")
    # print("ts data:", ts_data[:2, :, :5])
    # print("ts mask:", ts_mask[:2, :, :5])
    ts_data[masking] = 0
    ts_mask[masking] = 0
    ts_tau[masking] = 0
    # print("After Mask:")
    # print("ts data:", ts_data[:2, :, :5])
    # print("ts mask:", ts_mask[:2, :, :5])

    del masking, now_len, ts_len
    return ts_data, ts_mask, ts_tau


def binomial_masking(ts_data, ts_mask, ts_tau, p=0.05):
    '''
    :param ts_data: B L K
    :param ts_mask: B L K
    :param p: mask掉时间戳的概率
    :return:
    '''
    B, L, K = ts_data.shape
    masking = torch.from_numpy(np.random.binomial(1, p, size=(B, L))).to(torch.bool)
    masking = masking.unsqueeze(2).expand(B, L, K).to(ts_data.device)
    ts_data[masking] = 0
    ts_mask[masking] = 0
    ts_tau[masking] = 0
    del masking
    return ts_data, ts_mask, ts_tau

def its_agumentation_bank(ts_data, ts_tt, ts_mask, ts_tau, meta):
    agumentation_choice = ['Continuous masking', 'Noise addition']  # 'Binomial masking'
    choice = random.choice(agumentation_choice)
    # 1. 随机部分时刻全部mask
    if choice == 'Continuous masking':
        ts_data, ts_mask, ts_tau = continuous_masking(ts_data, ts_tt, ts_mask, ts_tau)

    # 2. 随机噪声
    elif choice == 'Noise addition':
        ts_data = noise_addition(ts_data, ts_mask, meta)

    # 3. Binomial: 伯努利分布选择时刻mask
    elif choice == 'Binomial masking':
        ts_data, ts_mask, ts_tau = binomial_masking(ts_data, ts_mask, ts_tau)

    return ts_data, ts_mask, ts_tau



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_value = 68
    random.seed(seed_value)

    dataset = PretrainDataset(data_path='../data/pretrain_48.pkl', device=device, note_length=1024)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pretrain_collate_fn)

    meta_file = '../data/metadata.json'
    with open(meta_file, 'r') as json_file:
        meta = json.load(json_file)

    for index, batch in enumerate(dataloader):
        name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask = batch
        print(index)
        its_agumentation_bank(ts_data, ts_tt, ts_mask, ts_tau, meta)
