import torch.nn as nn
import sys
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class ScaledDotProductAttention_bias_note(nn.Module):

    def __init__(self, d_model, n_head, d_k, d_v, temperature, dropout=0.2):
        super().__init__()

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.n_head = n_head

        self.fc = nn.Linear(d_v * n_head, d_model)
        self.dropout_output = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        q = rearrange(self.w_qs(q), 'b n (h d) -> b h n d', h=self.n_head)
        k = rearrange(self.w_ks(k), 'b n (h d) -> b h d n', h=self.n_head)
        v = rearrange(self.w_vs(v), 'b n (h d) -> b h n d', h=self.n_head)

        attn = torch.matmul(q, k) / self.temperature  # b h n n

        if mask is not None:
            if attn.dim() > mask.dim():
                mask = mask.unsqueeze(1).expand(attn.shape)
            attn = attn.masked_fill(mask, -1e4)  # TODO: 这个是不是不够小，但是再小就无法使用了

        attn = self.dropout(F.softmax(attn, dim=-1))  # b h n n

        v = torch.matmul(attn, v)  # b h n d

        v = rearrange(v, 'b h n d -> b n (h d)')

        v = self.dropout_output(self.fc(v))  # b n d

        return v, attn


class PositionwiseFeedForward_note(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x


PAD = 0

def get_note_attn_key_pad_mask_K(mask):
    """ For masking out the padding part of key sequence. """
    seq_q = rearrange(mask, 'b n -> b n 1')
    seq_k = rearrange(mask, 'b n -> b 1 n')
    padding_mask = torch.matmul(seq_q, seq_k).eq(PAD)
    return padding_mask

class NoteEncoderLayer(nn.Module):
    """ attention + FFNN """

    def __init__(self, args, d_model, n_head, d_k, d_v, dropout=0.1):
        super(NoteEncoderLayer, self).__init__()

        self.slf_tem_attn = ScaledDotProductAttention_bias_note(d_model, n_head, d_k, d_v, temperature=d_k ** 0.5, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward_note(d_model, d_model, dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, note_emb, note_mask):
        # [B, N, D] [B N]
        # 1. note attn mask
        note_attn_mask = get_note_attn_key_pad_mask_K(mask=note_mask)  # b n n

        # 2. attention
        output = self.layer_norm(note_emb)

        output, enc_attn = self.slf_tem_attn(output, output, output, mask=note_attn_mask)

        output_enc = output + note_emb  # b n d

        # FFFNN
        # residue = enc_output
        output = self.layer_norm(output_enc)

        output = self.pos_ffn(output)

        output = output + output_enc

        # optional
        output = self.layer_norm(output)

        return output, enc_attn