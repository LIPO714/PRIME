'''
mTand module
'''

import torch
import math
from torch import nn
import torch.nn.functional as F

'''
Q1: Query选择哪几个时刻？
肯定要选择非note时刻的，因为note部分也需要实现mTand功能
M1: 因为note是连续选择的，因此就选择note ts附近的随机几个时刻
M2: 再0-48这个时间窗口内，随机选择几个时刻
'''

'''
Q2: TS因为要编码，TS->Time Embedding, 因为需要计算时间匹配度，因此这个TS是否需要归一化?
1、check一下WarpFormer的做法。保持一致吧。应该需要/48
'''

class MultiTimeAttention(nn.Module):
    def __init__(self, embed_time=32, num_heads=8, input_dim=32, nhidden=32, dropout=0.1, value_transfer=True):
        super(MultiTimeAttention, self).__init__()
        # value transfer flag
        self.value_transfer = value_transfer
        if self.value_transfer:
            self.value_linears = nn.Sequential(nn.Linear(input_dim, nhidden),
                                               nn.LayerNorm(nhidden),  # TODO:New
                                               nn.ReLU(),
                                               nn.Linear(nhidden, input_dim))
            self.decay = nn.Linear(1, 1, bias=False)

        # time emb
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

        # time emb linear
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim  # value_dim
        self.nhidden = nhidden  # output dim
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim * num_heads, nhidden)])

        self.dropout=dropout

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)  # B L 1
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def value_transfer_embedding(self, value, d_t):
        # value B L_t L D
        # d_t B L_t L
        # Condition analysis
        d_value = self.value_linears(value)
        d_t = d_t.unsqueeze(-1)
        time_decay = 1 / (1 + self.decay(d_t))
        value = (d_value * time_decay) * d_t + value
        return value

    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"

        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)  # B H L_t L
        # print(scores.shape)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)  # B H L_t L dim  最后一维复制
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(-1)  # B 1 L 1

            # print(mask.shape)
            # print(scores.shape)
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)
        p_attn = F.softmax(scores, dim=-2)  # B H L_t L dim  在L维度上归一化
        if self.dropout is not None:
            p_attn = F.dropout(p_attn, p=self.dropout, training=self.training)
        #             p_attn = dropout(p_attn)

        # print(p_attn)
        # print(p_attn.shape)
        # print(value.unsqueeze(1))
        # print(value.unsqueeze(1).shape)
        # print(p_attn * value.unsqueeze(1))
        return torch.sum(p_attn * value.unsqueeze(1), -2), p_attn  # B H L_t L dim * B 1 1 L dim; 然后在L维上求和-> B H L_t dim

    def forward(self, query_tt, key_tt, value, emb_mask=None):
        # query tt: B L_t
        # key tt: B L
        # value: B L D
        # emb mask: B L

        # TODO:check each step

        L_t = query_tt.shape[1]

        # transfer(value)
        if self.value_transfer:
            value = value.unsqueeze(1).repeat_interleave(L_t, dim=1)  # B L_t L D
            d_t = query_tt.unsqueeze(2) - key_tt.unsqueeze(1).expand(-1, L_t, -1)
            value = self.value_transfer_embedding(value, d_t)

        else:
            value = value.unsqueeze(1)  # B 1 L D

        # 1. tt->time emb
        query_tt_emb = self.learn_time_embedding(query_tt)
        key_tt_emb = self.learn_time_embedding(key_tt)

        # 2. time emb->linear->split to n head
        batch, _, seq_len, dim = value.shape
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query_tt_emb, key_tt_emb))]  # B H L E/D

        # 3. attention mask  B L -> B 1 L
        if emb_mask is not None:
            # Same mask applied to all h heads.
            emb_mask = emb_mask.unsqueeze(1)  # B 1 L

        # 4. attention
        x, _ = self.attention(query, key, value, emb_mask)
        x = x.transpose(1, 2).contiguous() \
            .view(batch, -1, self.h * dim)
        return self.linears[-1](x)