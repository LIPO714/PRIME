import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class AggregationAttention(nn.Module):
    def __init__(self, input_dim, attention_dim=128):
        super(AggregationAttention, self).__init__()
        self.attention_layer = nn.Linear(input_dim, attention_dim)
        self.context_vector = nn.Parameter(torch.randn(attention_dim, 1))

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.size()

        # Compute the attention scores
        attention_scores = self.attention_layer(x)  # (B, L, attention_dim)
        attention_scores = torch.tanh(attention_scores)  # (B, L, attention_dim)

        # Compute the weights
        weights = torch.matmul(attention_scores, self.context_vector)  # (B, L, 1)
        weights = weights.squeeze(-1)  # (B, L)
        weights = F.softmax(weights, dim=1)  # (B, L)

        # Compute the weighted sum of the inputs
        weights = weights.unsqueeze(-1)  # (B, L, 1)
        aggregated_output = torch.sum(x * weights, dim=1)  # (B, D)

        return aggregated_output

class EASY_ClassificationHead(nn.Module):
    def __init__(self, args, device):
        super(EASY_ClassificationHead, self).__init__()
        if args.task == "48ihm":
            self.n_classes = 1
        elif args.task == "24pheno":
            self.n_classes = 25
        self.emb_dim = args.ts_embed_dim

        self.dropout = args.dropout

        self.ts_agg = AggregationAttention(self.emb_dim, self.emb_dim)
        self.note_agg = AggregationAttention(self.emb_dim, self.emb_dim)

        self.out_layer = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.n_classes)
        )

    def forward(self, ts_rep, note_rep):

        ts_rep = self.ts_agg(ts_rep)
        note_rep = self.note_agg(note_rep)

        return self.out_layer(torch.cat([ts_rep, note_rep], dim=1))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        x = self.self_attn(x, x, x)[0]
        x = self.layer_norm1(x + res)

        res = x
        x = self.linear2(torch.relu(self.linear1(x)))
        x = self.layer_norm2(x + res)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, args, device):
        super(ClassificationHead, self).__init__()
        if args.task == "48ihm":
            self.n_classes = 1
            q_seq_len = 49
        elif args.task == "24pheno":
            self.n_classes = 25
            q_seq_len = 25
        self.emb_dim = args.ts_embed_dim
        self.nhead = args.classifier_nhead
        self.layers = args.classifier_layers
        self.dropout = args.dropout
        # self.ts_decoder_list = nn.ModuleList(
        #     [DecoderLayer(self.emb_dim, self.nhead) for _ in range(self.layers)]
        # )
        # self.note_decoder_list = nn.ModuleList(
        #     [DecoderLayer(self.emb_dim, self.nhead) for _ in range(self.layers)]
        # )
        self.self_cross_module = TransformerCrossEncoder(embed_dim=self.emb_dim,
                                        num_heads=self.nhead,
                                        layers=self.layers,
                                        device=device,
                                        attn_dropout=self.dropout,
                                        relu_dropout=self.dropout,
                                        res_dropout=self.dropout,
                                        embed_dropout=self.dropout,
                                        q_seq_len_1=q_seq_len)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, self.emb_dim))
        self.fc = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.emb_dim * 2, self.emb_dim * 2)
        )
        self.out_layer = nn.Linear(self.emb_dim * 2, self.n_classes)

    def forward(self, ts_rep, note_rep):

        # for layer in range(self.layers):
        #     ts_rep = self.ts_decoder_list[layer](ts_rep)
        #     note_rep = self.note_decoder_list[layer](note_rep)
        # ts_rep = self.avg_pool(ts_rep.permute(0, 2, 1)).squeeze(1)
        # note_rep = self.avg_pool(note_rep.permute(0, 2, 1)).squeeze(1)
        # x = torch.cat((ts_rep, note_rep), dim=1)

        # ts_rep, note_rep B L D
        ts_rep = ts_rep.transpose(0, 1)  # L B D
        note_rep = note_rep.transpose(0, 1)  # L B D

        # print("ts_rep.shape:", ts_rep.shape)
        # print("note_rep.shape:", note_rep.shape)

        hiddens = self.self_cross_module([note_rep, ts_rep])
        note_rep_with_ts, ts_rep_with_note = hiddens  # L B D
        last_hs = torch.cat([note_rep_with_ts[-1], ts_rep_with_note[-1]], dim=1)  # B 2D

        last_hs_fc = self.fc(last_hs)
        last_hs_fc += last_hs
        output = self.out_layer(last_hs_fc)
        return output



class TransformerCrossEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerCrossEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, device,attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False,q_seq_len_1=None,q_seq_len_2=None):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.device=device

        self.q_seq_len_1=q_seq_len_1
        self.q_seq_len_2=q_seq_len_2
        # self.intermediate=intermediate
        self.embed_positions_q_1=nn.Embedding(self.q_seq_len_1,embed_dim,padding_idx=0)
        nn.init.normal_(self.embed_positions_q_1.weight, std=0.02)

        if self.q_seq_len_2!= None:
            self.embed_positions_q_2=nn.Embedding(self.q_seq_len_2,embed_dim,padding_idx=0)
            nn.init.normal_(self.embed_positions_q_2.weight, std=0.02)

            self.embed_positions_q=nn.ModuleList([self.embed_positions_q_1,self.embed_positions_q_2])
        else:
            self.embed_positions_q=nn.ModuleList([self.embed_positions_q_1,self.embed_positions_q_1,])


        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerCrossEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])

    def forward(self, x_in_list):
        """
        Args:
            x_in_list (list of FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the list of last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`

        """

        # import pdb;
        # pdb.set_trace()
        x_list=x_in_list
        length_x1 = x_list[0].size(0) # (length,Batch_size,input_dim)
        length_x2 = x_list[1].size(0)
        x_list = [ self.embed_scale * x_in for x_in in x_in_list]
        if self.q_seq_len_1 is not None:
            position_x1 = torch.tensor(torch.arange(length_x1),dtype=torch.long).to(self.device)
            position_x2 = torch.tensor(torch.arange(length_x2),dtype=torch.long).to(self.device)
            positions=[position_x1 ,position_x2]
            x_list=[ l(position_x).unsqueeze(0).transpose(0,1) +x for l, x,position_x in zip(self.embed_positions_q, x_list,positions)]
              # Add positional embedding
        x_list[0]=F.dropout(x_list[0], p=self.dropout)
        x_list[1]=F.dropout(x_list[1], p=self.dropout)

        # encoder layers

        # x_low_level=None


        for layer in self.layers:
            x_list= layer(x_list) #proj_x_txt, proj_x_ts


        if self.normalize:
            x_list=[ l(x)  for l, x in zip(self.layer_norm, x_list)]
        return x_list




class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                     attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pre_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

        self.self_attns = nn.ModuleList([MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        ) for _ in range(2)])

        # self.post_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

        self.pre_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

        self.cross_attn_1 = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.cross_attn_2 = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        # self.post_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.pre_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])
        self.fc1 = nn.ModuleList([nn.Linear(self.embed_dim, 4*self.embed_dim) for _ in range(2)])  # The "Add & Norm" part in the paper
        self.fc2 = nn.ModuleList([nn.Linear(4*self.embed_dim, self.embed_dim) for _ in range(2)])
        # self.post_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])


    def forward(self, x_list):
        """
        Args:
            x (List of Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            list of encoded output of shape `(batch, src_len, embed_dim)`
        """
        ###self attn
        residual = x_list

        x_list = [l(x) for l, x in zip(self.pre_self_attn_layer_norm, x_list)]

        output= [l(query=x, key=x, value=x) for l, x in zip(self.self_attns, x_list)]

        x_list=[ x for x, _ in output]

        x_list[0]=F.dropout(x_list[0], p=self.res_dropout)
        x_list[1]=F.dropout(x_list[1], p=self.res_dropout)

        x_list = [r + x  for r, x in zip(residual, x_list) ]
#         x_list = [l(x) for l, x in zip(self.post_self_attn_layer_norm, x_list)]

        #### cross attn

        residual=x_list
        x_list = [l(x) for l, x in zip(self.pre_encoder_attn_layer_norm, x_list)]
        x_txt,x_ts=  x_list #proj_x_txt, proj_x_ts

        # cross: ts -> txt
        x_ts_to_txt,_=self.cross_attn_1(query=x_txt, key=x_ts, value=x_ts)
        # cross:  txt->ts
        x_txt_to_ts,_=self.cross_attn_2(query=x_ts, key=x_txt, value=x_txt)

        # else:
        #     x_low_level = [l(x) for l, x in zip(self.pre_encoder_attn_layer_norm, x_low_level)]
        #     x_txt_low,x_ts_low=  x_low_level
        #     # cross: ts -> txt
        #     x_ts_to_txt,_=self.cross_attn_1(query=x_txt, key=x_ts_low, value=x_ts_low)
        #     # cross:  txt->ts
        #     x_txt_to_ts,_=self.cross_attn_2(query=x_ts, key=x_txt_low, value=x_txt_low)


        x_ts_to_txt = F.dropout(x_ts_to_txt, p=self.res_dropout)
        x_txt_to_ts = F.dropout(x_txt_to_ts, p=self.res_dropout)

        x_list = [r+ x for r, x in zip(residual, (x_ts_to_txt, x_txt_to_ts))]

#         x_list = [l(x) for l, x in zip(self.post_encoder_attn_layer_norm, x_list)]

        # FNN
        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_ffn_layer_norm, x_list)]
        x_list = [F.relu(l(x)) for l, x in zip(self.fc1, x_list)]

        x_list[0]=F.dropout(x_list[0], p=self.relu_dropout)
        x_list[1]=F.dropout(x_list[1], p=self.relu_dropout)

        x_list = [l(x) for l, x in zip(self.fc2, x_list)]

        x_list[0]=F.dropout(x_list[0], p=self.res_dropout)
        x_list[1]=F.dropout(x_list[1], p=self.res_dropout)

        x_list = [r + x for r, x in zip(residual, x_list)]

#         x_list = [l(x) for l, x in zip(self.post_ffn_layer_norm, x_list)]

        return x_list


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        # import pdb;
        # pdb.set_trace()
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
