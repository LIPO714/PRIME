import random

import torch


def demogra_agumentation_bank(d, meta):
    # agumentation_choice = ['Noise']
    # choice = random.choice(agumentation_choice)
    # 1. 给年龄加噪
    noise_percent = 0.05
    b, _ = d.shape
    noise = (torch.randn(b) * noise_percent).to(d.device)
    # print(f'Before:{d[:, 0]}')
    d[:, 0] = d[:, 0] + noise
    # print(f'After:{d[:, 0]}')

    del noise
    return d