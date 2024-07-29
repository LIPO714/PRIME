import torch
from torch import nn

from loss import CONT_Loss, RESTORE_LOSS
from model.TS_dual_attention import EncoderLayer, Attention_Aggregator
from model.encoder import Time_Encoder, Note_Tau_Encoder, Note_Time_Encoder, Var_Encoder, Type_Encoder, \
    Density_Encoder, Invar_Encoder, Node_Encoder, Cate_Encoder
from einops import rearrange, repeat

from model.mTand import MultiTimeAttention
from model.mFand import MultiFeatureAttention
from model.module import gateMLP
from model.note_attention import NoteEncoderLayer


class BertForRepresentation(nn.Module):

    def __init__(self, args, BioBert):
        super().__init__()
        self.bert = BioBert

        self.dropout = torch.nn.Dropout(BioBert.config.hidden_dropout_prob)
        self.model_name = args.model_name
        self.bert_rep = args.bert_rep

    def forward(self, note_token_id, note_attention_mask):
        # note_token_id: B 5 T
        # note_attention_mask: B 5 T

        B = len(note_token_id)

        txt_arr = []

        for input_ids, attention_mask in zip(note_token_id, note_attention_mask):
            # reshape: N T
            # 1. input to bert
            if 'Longformer' in self.model_name:

                attention_mask[attention_mask==1] += 1
                text_embeddings = self.bert(input_ids, attention_mask=attention_mask)
            else:
                text_embeddings = self.bert(input_ids, attention_mask=attention_mask)

            # add to list
            if self.bert_rep == "CLS":
                text_embeddings = text_embeddings[0][:, 0, :]
            elif self.bert_rep == "Avg-pooling":
                text_embeddings = text_embeddings[0]

                attention_mask[attention_mask>0] = 1

                token_num = torch.sum(attention_mask, dim=-1)  # L
                token_num[token_num == 0] = 1

                attention_mask = attention_mask.unsqueeze(-1).expand_as(text_embeddings)
                text_embeddings = text_embeddings * attention_mask  # L token 768
                text_embeddings = torch.sum(text_embeddings, dim=1) / token_num.unsqueeze(1)  # L 768

            text_embeddings = self.dropout(text_embeddings)
            txt_arr.append(text_embeddings)
        # output process

        del note_token_id, note_attention_mask, attention_mask
        # B*N, T
        # 3. stack
        txt_arr = torch.stack(txt_arr)
        # print("txt_arr.shape:", txt_arr.shape)  # [2, 3, 768]
        return txt_arr


class TextModel(nn.Module):
    def __init__(self,args,Biobert=None):
        """
        Construct a TextModel.
        """
        super(TextModel, self).__init__()

        self.args = args
        self.bertrep=BertForRepresentation(args,Biobert)
        self.note_rep = args.note_rep

        self.rep_encoder = nn.Linear(768, args.embed_dim)
        if args.note_rep == "val+time" or args.note_rep == "val+time+tau":
            self.time_encoder = Note_Time_Encoder(embed_time=args.embed_dim)
        if args.note_rep == "val+time+tau":
            self.tau_encoder = Note_Tau_Encoder(embed_time=args.embed_dim)

        self.note_attention = NoteEncoderLayer(args, args.embed_dim, args.note_n_head, args.note_d_k, args.note_d_v, args.dropout)

        self.linear = nn.Linear(args.embed_dim, args.ts_embed_dim)


    def forward(self, note_token_id, note_attention_mask, note_mask, note_tt, note_tau, restore_index=None):
        """
        dimension [batch_size, seq_len, n_features]
        """
        x_txt = self.bertrep(note_token_id, note_attention_mask)  # B N 768

        txt_emb = self.rep_encoder(x_txt)  # B N emb
        if self.note_rep == "val":
            note_emb = txt_emb
        elif self.note_rep == "val+time":
            time_emb = self.time_encoder(note_tt)
            note_emb = txt_emb + time_emb
        elif self.note_rep == "val+time+tau":
            time_emb = self.time_encoder(note_tt)
            tau_emb = self.tau_encoder(note_tau)
            note_emb = txt_emb + time_emb + tau_emb  # B N Emb

        note_emb = note_emb * rearrange(note_mask, 'b n -> b n 1')

        if restore_index != None and self.args.task == "pretrain":
            B, N, E = note_emb.shape

            new_note_emb = torch.zeros((B, N-1, E), dtype=note_emb.dtype).to(note_tt.device)
            new_note_mask = torch.zeros((B, N-1), dtype=note_mask.dtype).to(note_tt.device)
            new_note_tt = torch.zeros((B, N-1), dtype=note_tt.dtype).to(note_tt.device)

            restore_note_emb = torch.zeros((B, E), dtype=note_emb.dtype).to(note_tt.device)
            for i in range(B):
                delete_index = restore_index[i]
                restore_note_emb[i] = note_emb[i, delete_index, :]
                if delete_index == 0:
                    new_note_emb[i] = note_emb[i, 1:]
                    new_note_mask[i] = note_mask[i, 1:]
                    new_note_tt[i] = note_tt[i, 1:]
                elif delete_index == N-1:
                    new_note_emb[i] = note_emb[i, :-1]
                    new_note_mask[i] = note_mask[i, :-1]
                    new_note_tt[i] = note_tt[i, :-1]
                else:
                    new_note_emb[i] = torch.cat((note_emb[i, :delete_index], note_emb[i, delete_index+1:]), dim=0)
                    new_note_mask[i] = torch.cat((note_mask[i, :delete_index], note_mask[i, delete_index+1:]), dim=0)
                    new_note_tt[i] = torch.cat((note_tt[i, :delete_index], note_tt[i, delete_index+1:]), dim=0)

            note_emb = new_note_emb
            note_mask = new_note_mask
            note_tt = new_note_tt

        # attention & mask
        note_emb, enc_attn = self.note_attention(note_emb, note_mask)

        note_emb = self.linear(note_emb)
        if restore_index != None and self.args.task == "pretrain":
            restore_note_emb = self.linear(restore_note_emb)
        else:
            restore_note_emb = None

        return note_emb, note_mask, note_tt, restore_note_emb  # [B N-1 E] [B N-1] [B N-1] [B E]


class TSModel(nn.Module):
    def __init__(self,args):
        """
        Construct a TextModel.
        """
        super(TSModel, self).__init__()

        self.args = args

        self.embed_dim = args.ts_dim
        self.ts_embed_dim = args.ts_embed_dim
        self.var_dim = args.var_dim
        self.invar_dim = args.invar_dim

        # continue and categorical
        self.cate_dim = args.cate_dim
        self.continue_dim = args.var_dim - self.cate_dim
        cate_type_str = args.cate_type
        cate_type_list = cate_type_str.split("_")
        self.cate_type = []
        for i in cate_type_list:
            self.cate_type.append(int(i))

        # Invar Encoder
        self.invar_enc = Invar_Encoder(out_dim=self.embed_dim, num_type=self.invar_dim)

        self.ts_model_backbone = args.ts_model_backbone
        # encoder
        if args.ts_model_backbone == "DualAttention":
            # var encoder
            self.var_enc = Var_Encoder(output_dim=self.embed_dim, num_type=self.continue_dim)
            # cate encoder
            if self.cate_dim > 0:
                self.cate_enc = Cate_Encoder(output_dim=self.embed_dim, num_type=self.cate_dim, embedding_type=self.cate_type)
            # type encoder
            self.type_enc = Type_Encoder(d_model=self.embed_dim, num_types=self.var_dim)
            self.type_matrix = torch.tensor([int(i) for i in range(1, self.var_dim + 1)]).to(torch.int)
            self.type_matrix = rearrange(self.type_matrix, 'k -> 1 1 k')
            # time encoder
            self.learn_time_embedding = Time_Encoder(self.embed_dim, self.var_dim)
            # density encoder
            self.density_encoder = Density_Encoder(self.embed_dim, self.var_dim)

            # node encoder
            self.node_encoder = Node_Encoder(args.remove_rep, self.embed_dim, self.embed_dim)

            # dual attention
            self.dual_attention_stack = nn.ModuleList([
                EncoderLayer(args=args, d_model=self.embed_dim, d_inner=int(self.embed_dim/2), n_head=args.n_head,\
                             d_k=args.d_k, d_v=args.d_v, dropout=args.dropout)
                for _ in range(args.ts_dual_attention_layer)
            ])

            # agg_attn
            self.agg_attention = Attention_Aggregator(self.embed_dim, self.ts_embed_dim)

        elif args.ts_model_backbone == "Ori":
            pass


    def forward(self, demogra, ts_data, ts_tt, ts_mask, ts_tau, restore_index=None):
        # ts data: B L K
        # ts_tt: B L
        # ts_mask: B L K
        # ts_tau: B L
        B, L, K = ts_data.shape

        # invar encoder  [B, num_invar] -> [B, D]
        invar_emb = self.invar_enc(demogra)

        if self.ts_model_backbone == "DualAttention":
            # continue var emb
            var_emb = self.var_enc(ts_data[:, :, :self.continue_dim], ts_mask[:, :, :self.continue_dim])
            var_emb = rearrange(var_emb, 'b l k d -> b k l d')  # [B,K-cate,L,D]

            # cate var emb
            if self.cate_dim > 0:
                cate_emb = self.cate_enc(ts_data[:, :, self.continue_dim:], ts_mask[:, :, self.continue_dim:])
                cate_emb = rearrange(cate_emb, 'b l k d -> b k l d')  # [B,cate,L,D]

                var_emb = torch.cat([var_emb, cate_emb], dim=1)  # B K L D

            # type emb
            type_emb = self.type_matrix.to(ts_data.device)  # 1 1 K
            type_emb = self.type_enc(type_emb)  # 1 1 K D
            type_emb = rearrange(type_emb, 'b l k d -> b k l d')  # [B,K,L,D]

            # time emb
            time_enc_k = self.learn_time_embedding(ts_tt, ts_mask)  # [B,L], [B,L,K]-->[B,L,K,D]
            time_enc_k = rearrange(time_enc_k, 'b l k d -> b k l d')  # [B,K,L,D]

            # density emb
            density_emb = self.density_encoder(ts_tau, ts_mask)
            density_emb = rearrange(density_emb, 'b l k d -> b k l d')

            # invar
            node_emb = self.node_encoder(var_emb, density_emb, type_emb, time_enc_k, invar_emb, ts_mask)

            if self.args.remove_rep == 'time':
                h = density_emb + node_emb + type_emb
            elif self.args.remove_rep == 'type':
                h = density_emb + time_enc_k + node_emb
            elif self.args.remove_rep == 'density':
                h = time_enc_k + node_emb + type_emb
            elif self.args.remove_rep == 'invar':
                h = density_emb + time_enc_k + type_emb
            else:
                h = density_emb + time_enc_k + node_emb + type_emb

            h0 = var_emb + h
            ts_mask = rearrange(ts_mask, 'b l k -> b k l')


            ts_emb_mask = torch.sum(ts_mask, dim=1)  # B L-l
            ts_emb_mask[ts_emb_mask>1] = 1

            # dual attention + agg
            z0 = None
            for i, dual_attention in enumerate(self.dual_attention_stack):
                if i > 0 and self.args.full_attn:
                    ts_mask = torch.ones_like(ts_mask).to(ts_mask.device)

                h0, _, _ = dual_attention(h0, ts_mask)

                output = self.agg_attention(h0, rearrange(ts_mask, 'b k l -> b k l 1'))  # [B K L D] --> [B L D]

                if z0 is not None and z0.shape == output.shape:
                    z0 = z0 + output
                else:
                    z0 = output

            # print("z0.shape:", z0.shape)

        elif self.ts_model_backbone == "Ori":
            z0 = torch.cat([ts_data, ts_tau, ts_mask], dim=2)
            ts_emb_mask = torch.cat([ts_mask, ts_mask, ts_mask], dim=2)

        return z0, ts_emb_mask, ts_tt, invar_emb  # z0 [B L-l D]/[B L-l K] ts_emb_mask [B L-l]/[B L-l 3*K] ts_tt [B L-l]


class ImputeTSEncoder(nn.Module):
    def __init__(self,args):
        """
        Construct a Impute ts Encoder
        """
        super(ImputeTSEncoder, self).__init__()

        self.args = args

        self.ts_embed_dim = args.ts_embed_dim
        self.var_dim = args.var_dim

        # encoder
        # var encoder
        self.var_enc = nn.Sequential(
                nn.Linear(args.var_dim, args.ts_embed_dim),
                nn.LayerNorm(args.ts_embed_dim),
                nn.ReLU(),
                nn.Linear(args.ts_embed_dim, args.ts_embed_dim),
            )

    def forward(self, invar_emb, query_ts_data, query_ts_tt):
        # invar_emb: B D
        # query_ts_data: B L_t+l K
        # query_ts_tt: B L_t+l

        # var emb
        var_emb = self.var_enc(query_ts_data)  # B L_t+l D

        # time emb
        # time_enc_k = self.learn_time_embedding(query_ts_tt)  # [B L_t+l]-->[B L_t+l,D]

        # invar emb
        # invar_emb = invar_emb.unsqueeze(1)  # B D -> B 1 D

        query_ts_rep = var_emb  # B L_t+l D

        return query_ts_rep


class MULTCrossModel(nn.Module):
    def __init__(self,args,Biobert=None):
        """
        Construct a MulT Cross model.
        """
        super(MULTCrossModel, self).__init__()

        self.TextModel = TextModel(args, Biobert=Biobert)
        self.TSModel = TSModel(args)

        self.task = args.task
        self.mode = args.mode
        self.ts_backbone = args.ts_model_backbone

        if args.ts_model_backbone == "DualAttention":
            ts_input_dim = args.ts_embed_dim
            ts_input_value_dim = args.var_dim * 2
        elif args.ts_model_backbone == "Ori":
            ts_input_dim = args.var_dim * 3
            ts_input_value_dim = args.var_dim * 2

        self.ts_mtand = MultiTimeAttention(embed_time=args.ts_dim,
                                           num_heads=8,
                                           input_dim=ts_input_dim,
                                           nhidden=args.ts_embed_dim,
                                           dropout=args.dropout,
                                           value_transfer=args.ts_mtand_value_transfer)
        self.ts_mfand = MultiFeatureAttention(embed_value=args.ts_dim,
                                            num_heads=8,
                                            input_value_dim=ts_input_value_dim,
                                            nhidden=args.ts_embed_dim,
                                            input_dim=ts_input_dim,
                                            dropout=args.dropout
                                            )
        self.note_mtand = MultiTimeAttention(embed_time=args.ts_dim,
                                             num_heads=8,
                                             input_dim=args.ts_embed_dim,
                                             nhidden=args.ts_embed_dim,
                                             dropout=args.dropout,
                                             value_transfer=args.note_mtand_value_transfer)

        # contrastive
        self.contrastive_num = args.contrastive_num

        self.ts_model_without_part = args.ts_model_wo_part
        self.mixup_level = args.mixup_level
        self.mtand_mfand_mixup_level = args.mtand_mfand_mixup_level

        if self.mtand_mfand_mixup_level == "batch_seq":
            self.mtand_mfand_moe = gateMLP(input_dim=args.ts_embed_dim * 2, hidden_size=args.ts_embed_dim, output_dim=1,  # args.ts_embed_dim,  # TODO: 1
                                            dropout=args.dropout)
        elif self.mtand_mfand_mixup_level == "batch_seq_feature":
            self.mtand_mfand_moe = gateMLP(input_dim=args.ts_embed_dim * 2, hidden_size=args.ts_embed_dim, output_dim=args.ts_embed_dim,
                                            dropout=args.dropout)

        if self.mixup_level == 'batch':
            self.moe = gateMLP(input_dim=args.ts_embed_dim * 2, hidden_size=args.ts_embed_dim, output_dim=1,
                               dropout=args.dropout)
        elif self.mixup_level == 'batch_seq':
            self.moe = gateMLP(input_dim=args.ts_embed_dim * 2, hidden_size=args.ts_embed_dim, output_dim=1,
                               dropout=args.dropout)
        elif self.mixup_level == 'batch_seq_feature':
            self.moe = gateMLP(input_dim=args.ts_embed_dim * 2, hidden_size=args.ts_embed_dim,
                               output_dim=args.ts_embed_dim,
                               dropout=args.dropout)
        else:
            raise ValueError("Unknown mixedup type")

        # self.ts_agg_emb_enc = nn.Sequential(
        #     nn.Linear(args.ts_embed_dim, args.ts_embed_dim),
        #     nn.LayerNorm(args.ts_embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(args.ts_embed_dim, args.ts_embed_dim),
        # )

        # loss
        self.loss_type = args.loss
        if args.loss == "CONT+RESTORE":
            self.alpha_emb_dim = int(args.alpha * args.ts_embed_dim)  # ts 0.5 *
            self.beta_emb_dim = int(args.beta * args.ts_embed_dim)  # note
            self.linear_decompos = args.linear_decompos
            if self.linear_decompos:
                self.projecter = nn.Linear(args.ts_embed_dim, args.ts_embed_dim)
            else:
                if self.alpha_emb_dim == self.beta_emb_dim:
                    pass
                elif self.beta_emb_dim > self.alpha_emb_dim:
                    self.cont_W = nn.Linear(self.alpha_emb_dim, self.beta_emb_dim, bias=False)
                elif self.beta_emb_dim < self.alpha_emb_dim:
                    self.cont_W = nn.Linear(self.beta_emb_dim, self.alpha_emb_dim, bias=False)

            self.cont_loss = CONT_Loss(world_size=args.world_size,
                                       temperature=args.temp,
                                       learnable_temp=args.learnable_temp)

            self.ts_decoder = nn.Sequential(
                nn.Linear(args.ts_embed_dim, args.ts_embed_dim),
                nn.LayerNorm(args.ts_embed_dim),
                nn.ReLU(),
                nn.Linear(args.ts_embed_dim, args.var_dim, bias=False),
            )
            self.note_decoder = nn.Sequential(
                nn.Linear(args.ts_embed_dim, args.ts_embed_dim),
                nn.ReLU(),
                nn.Linear(args.ts_embed_dim, args.ts_embed_dim, bias=False),
            )

            self.rest_loss = RESTORE_LOSS()

            self.lambda_1 = args.lambda_1
            self.lambda_2 = args.lambda_2


        elif args.loss == "CONT":
            self.projecter = nn.Linear(args.ts_embed_dim, int(args.ts_embed_dim/2))
            self.cont_loss = CONT_Loss(world_size=args.world_size,
                                       temperature=args.temp,
                                       learnable_temp=args.learnable_temp)

        elif args.loss == "RESTORE":
            self.ts_decoder = nn.Sequential(
                nn.Linear(args.ts_embed_dim, args.ts_embed_dim),
                nn.ReLU(),
                nn.Linear(args.ts_embed_dim, args.var_dim, bias=False),
            )
            self.note_decoder = nn.Sequential(
                nn.Linear(args.ts_embed_dim, args.ts_embed_dim),
                nn.ReLU(),
                nn.Linear(args.ts_embed_dim, args.ts_embed_dim, bias=False),
            )
            self.rest_loss = RESTORE_LOSS()

            self.lambda_2 = args.lambda_2

        # 具体任务的query time
        if args.task == "48ihm":
            self.query_tt = torch.arange(0, 49, 2).float().unsqueeze(0)
        elif args.task == "24pheno":
            self.query_tt = torch.arange(0, 25, 2).float().unsqueeze(0)

    def choose_query_tt(self, ts_tt, note_tt):
        B, _ = ts_tt.shape
        ts_min, _ = torch.min(ts_tt, dim=1)
        ts_max, _ = torch.max(ts_tt, dim=1)
        note_min, _ = torch.min(note_tt, dim=1)
        note_max, _ = torch.max(note_tt, dim=1)
        query_tt = torch.zeros((B, self.contrastive_num), dtype=ts_tt.dtype).to(ts_tt.device)
        for i in range(B):
            if ts_min[i] < note_min[i] and ts_max[i] > note_max[i]:
                min = note_min[i]
                max = note_max[i]
            else:
                min = ts_min[i]
                max = ts_max[i]
            query_tt[i] = (max - min) * torch.rand((self.contrastive_num), dtype=ts_tt.dtype).to(ts_tt.device) + min

        query_tt, _ = torch.sort(query_tt, dim=1)
        return query_tt

    def forward(self, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_token_id, note_attention_mask, note_tt, note_tau, note_mask, restore_index=None, restore_ts_label=None, restore_ts_mask=None, query_ts_tt=None, query_note_tt=None, query_ts_data=None, query_ts_mask=None):
        if self.task == "pretrain":
            ts_rep, ts_emb_mask, ts_rep_tt, invar_emb = self.TSModel(demogra, ts_data, ts_tt, ts_mask, ts_tau)

            if self.loss_type == "CONT+RESTORE" or self.loss_type == "RESTORE":
                ts_query_tt = query_ts_tt.clone()  # [B, L_t+l]
            else:
                ts_query_tt = query_ts_tt[:, :self.contrastive_num]  # [B, L_t]

            # mtand
            ts_mtand_agg_rep = self.ts_mtand(ts_query_tt, ts_rep_tt, ts_rep, ts_emb_mask)  #
            # mfand
            ts_query_value = torch.cat([query_ts_data, query_ts_mask], dim=2)
            ts_key_value = torch.cat([ts_data, ts_mask], dim=2)
            ts_mfand_agg_rep, ts_impute_rep = self.ts_mfand(ts_query_value, ts_key_value, ts_rep, ts_emb_mask, query_ts_data)

            if self.ts_model_without_part == "None":
                mtand_mfand_moe_gate = torch.cat([ts_mtand_agg_rep, ts_mfand_agg_rep], dim=-1)
                mtand_mfand_mixup_rate = self.mtand_mfand_moe(mtand_mfand_moe_gate)
                ts_agg_rep = mtand_mfand_mixup_rate * ts_mtand_agg_rep + (1 - mtand_mfand_mixup_rate) * ts_mfand_agg_rep

                if self.mixup_level == 'batch':
                    g_time = torch.max(ts_agg_rep, dim=0).values
                    g_value = torch.max(ts_impute_rep, dim=0).values
                    moe_gate = torch.cat([g_time, g_value], dim=-1)  # L_t+l, 2D
                elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
                    moe_gate = torch.cat([ts_agg_rep, ts_impute_rep], dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")
                mixup_rate = self.moe(moe_gate)
                ts_agg_rep = mixup_rate * ts_agg_rep + (1 - mixup_rate) * ts_impute_rep

            elif self.ts_model_without_part == "mTAND":
                ts_agg_rep = ts_mfand_agg_rep

                if self.mixup_level == 'batch':
                    g_time = torch.max(ts_agg_rep, dim=0).values
                    g_value = torch.max(ts_impute_rep, dim=0).values
                    moe_gate = torch.cat([g_time, g_value], dim=-1)  # L_t+l, 2D
                elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
                    moe_gate = torch.cat([ts_agg_rep, ts_impute_rep], dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")
                mixup_rate = self.moe(moe_gate)
                ts_agg_rep = mixup_rate * ts_agg_rep + (1 - mixup_rate) * ts_impute_rep

            elif self.ts_model_without_part == "mFAND":
                ts_agg_rep = ts_mtand_agg_rep

                if self.mixup_level == 'batch':
                    g_time = torch.max(ts_agg_rep, dim=0).values
                    g_value = torch.max(ts_impute_rep, dim=0).values
                    moe_gate = torch.cat([g_time, g_value], dim=-1)  # L_t+l, 2D
                elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
                    moe_gate = torch.cat([ts_agg_rep, ts_impute_rep], dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")
                mixup_rate = self.moe(moe_gate)
                ts_agg_rep = mixup_rate * ts_agg_rep + (1 - mixup_rate) * ts_impute_rep

            elif self.ts_model_without_part == "Impute":
                mtand_mfand_moe_gate = torch.cat([ts_mtand_agg_rep, ts_mfand_agg_rep], dim=-1)
                mtand_mfand_mixup_rate = self.mtand_mfand_moe(mtand_mfand_moe_gate)
                ts_agg_rep = mtand_mfand_mixup_rate * ts_mtand_agg_rep + (1 - mtand_mfand_mixup_rate) * ts_mfand_agg_rep

            elif self.ts_model_without_part == "feature_gate":
                ts_agg_rep = ts_mtand_agg_rep + ts_mfand_agg_rep + ts_impute_rep

            note_rep, note_emb_mask, note_rep_tt, restore_note_rep = self.TextModel(note_token_id, note_attention_mask, note_mask, note_tt, note_tau, restore_index[:, 2])

            if self.loss_type == "CONT+RESTORE" or self.loss_type == "RESTORE":
                note_query_tt = query_note_tt.clone()
            else:
                note_query_tt = query_note_tt[:, :self.contrastive_num]

            note_agg_rep = self.note_mtand(note_query_tt, note_rep_tt, note_rep, note_emb_mask)

            # 8. loss
            if self.loss_type == "CONT+RESTORE":
                if self.linear_decompos:
                    ts_cont_rep = ts_agg_rep[:, :self.contrastive_num]
                    note_cont_rep = note_agg_rep[:, :self.contrastive_num]

                    ts_cont_rep = self.projecter(ts_cont_rep)
                    note_cont_rep = self.projecter(note_cont_rep)

                else:
                    ts_cont_rep = ts_agg_rep[:, :self.contrastive_num, :self.alpha_emb_dim]
                    note_cont_rep = note_agg_rep[:, :self.contrastive_num, :self.beta_emb_dim]

                    if self.alpha_emb_dim == self.beta_emb_dim:
                        pass
                    elif self.beta_emb_dim > self.alpha_emb_dim:
                        ts_cont_rep = self.cont_W(ts_cont_rep)
                    elif self.beta_emb_dim < self.alpha_emb_dim:
                        note_cont_rep = self.cont_W(note_cont_rep)

                if torch.isnan(ts_cont_rep).any().item():
                    print("ts_cont_rep has nan")
                if torch.isnan(note_cont_rep).any().item():
                    print("note_cont_rep has nan")

                contrastive_loss = self.cont_loss(ts_cont_rep, note_cont_rep)

                ts_restore_rep = ts_agg_rep[:, self.contrastive_num:, :]
                note_restore_rep = note_agg_rep[:, self.contrastive_num:, :]

                if torch.isnan(ts_restore_rep).any().item():
                    print("ts_restore_rep has nan")
                if torch.isnan(note_cont_rep).any().item():
                    print("note_restore_rep has nan")

                ts_restore_rep = self.ts_decoder(ts_restore_rep)
                note_restore_rep = self.note_decoder(note_restore_rep)

                ts_restore_mse, note_restore_cos = self.rest_loss(ts_restore_rep, restore_ts_label, restore_ts_mask, note_restore_rep, restore_note_rep)

                loss = contrastive_loss + self.lambda_1 * ts_restore_mse + self.lambda_2 * note_restore_cos

                out = {
                    'loss': loss,
                    'contrastive_loss': contrastive_loss,
                    'ts_restore_mse': ts_restore_mse,
                    'note_restore_cos': note_restore_cos
                }


            elif self.loss_type == "CONT":

                ts_agg_rep = ts_agg_rep[:, :self.contrastive_num]
                note_agg_rep = note_agg_rep[:, :self.contrastive_num]
                ts_agg_rep = self.projecter(ts_agg_rep)
                note_agg_rep = self.projecter(note_agg_rep)
                loss = self.cont_loss(ts_agg_rep, note_agg_rep)

                out = {
                    'loss': loss,
                    'contrastive_loss': loss,
                    'ts_restore_mse': loss,
                    'note_restore_cos': loss
                }

            elif self.loss_type == "RESTORE":
                ts_restore_rep = ts_agg_rep[:, self.contrastive_num:, :]
                note_restore_rep = note_agg_rep[:, self.contrastive_num:, :]

                ts_restore_rep = self.ts_decoder(ts_restore_rep)
                note_restore_rep = self.note_decoder(note_restore_rep)

                ts_restore_mse, note_restore_cos = self.rest_loss(ts_restore_rep, restore_ts_label, restore_ts_mask, note_restore_rep, restore_note_rep)

                loss = ts_restore_mse + self.lambda_2 * note_restore_cos

                out = {
                    'loss': loss,
                    'contrastive_loss': loss,
                    'ts_restore_mse': ts_restore_mse,
                    'note_restore_cos': note_restore_cos
                }

            return loss, out

        else:  # ihm pheno
            ts_rep, ts_emb_mask, ts_rep_tt, invar_emb = self.TSModel(demogra, ts_data, ts_tt, ts_mask, ts_tau)

            ts_mtand_agg_rep = self.ts_mtand(query_ts_tt, ts_rep_tt, ts_rep, ts_emb_mask)  #
            # mfand
            ts_query_feature = torch.cat([query_ts_data, query_ts_mask], dim=2)
            ts_key_feature = torch.cat([ts_data, ts_mask], dim=2)
            ts_mfand_agg_rep, ts_impute_rep = self.ts_mfand(ts_query_feature, ts_key_feature, ts_rep, ts_emb_mask, query_ts_data)

            if self.ts_model_without_part == "None":
                mtand_mfand_moe_gate = torch.cat([ts_mtand_agg_rep, ts_mfand_agg_rep], dim=-1)
                mtand_mfand_mixup_rate = self.mtand_mfand_moe(mtand_mfand_moe_gate)
                ts_agg_rep = mtand_mfand_mixup_rate * ts_mtand_agg_rep + (1 - mtand_mfand_mixup_rate) * ts_mfand_agg_rep

                if self.mixup_level == 'batch':
                    g_time = torch.max(ts_agg_rep, dim=0).values
                    g_value = torch.max(ts_impute_rep, dim=0).values
                    moe_gate = torch.cat([g_time, g_value], dim=-1)  # L_t+l, 2D
                elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
                    moe_gate = torch.cat([ts_agg_rep, ts_impute_rep], dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")
                mixup_rate = self.moe(moe_gate)
                ts_agg_rep = mixup_rate * ts_agg_rep + (1 - mixup_rate) * ts_impute_rep

            elif self.ts_model_without_part == "mTAND":
                ts_agg_rep = ts_mfand_agg_rep

                if self.mixup_level == 'batch':
                    g_time = torch.max(ts_agg_rep, dim=0).values
                    g_value = torch.max(ts_impute_rep, dim=0).values
                    moe_gate = torch.cat([g_time, g_value], dim=-1)  # L_t+l, 2D
                elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
                    moe_gate = torch.cat([ts_agg_rep, ts_impute_rep], dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")
                mixup_rate = self.moe(moe_gate)
                ts_agg_rep = mixup_rate * ts_agg_rep + (1 - mixup_rate) * ts_impute_rep

            elif self.ts_model_without_part == "mFAND":
                ts_agg_rep = ts_mtand_agg_rep

                if self.mixup_level == 'batch':
                    g_time = torch.max(ts_agg_rep, dim=0).values
                    g_value = torch.max(ts_impute_rep, dim=0).values
                    moe_gate = torch.cat([g_time, g_value], dim=-1)  # L_t+l, 2D
                elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
                    moe_gate = torch.cat([ts_agg_rep, ts_impute_rep], dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")
                mixup_rate = self.moe(moe_gate)
                ts_agg_rep = mixup_rate * ts_agg_rep + (1 - mixup_rate) * ts_impute_rep

            elif self.ts_model_without_part == "Impute":
                mtand_mfand_moe_gate = torch.cat([ts_mtand_agg_rep, ts_mfand_agg_rep], dim=-1)
                mtand_mfand_mixup_rate = self.mtand_mfand_moe(mtand_mfand_moe_gate)
                ts_agg_rep = mtand_mfand_mixup_rate * ts_mtand_agg_rep + (1 - mtand_mfand_mixup_rate) * ts_mfand_agg_rep

            elif self.ts_model_without_part == "feature_gate":
                ts_agg_rep = ts_mtand_agg_rep + ts_mfand_agg_rep + ts_impute_rep

            note_rep, note_emb_mask, note_rep_tt, _ = self.TextModel(note_token_id, note_attention_mask, note_mask, note_tt, note_tau)

            note_agg_rep = self.note_mtand(query_note_tt, note_rep_tt, note_rep, note_emb_mask)

            return ts_agg_rep, note_agg_rep  # [B 48 D] [B 48 D]


    def eval_state(self, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_token_id, note_attention_mask, note_tt, note_tau, note_mask, restore_index=None, restore_ts_label=None, restore_ts_mask=None, query_ts_tt=None, query_note_tt=None, query_ts_data=None, query_ts_mask=None):

        ts_rep, ts_emb_mask, ts_rep_tt, invar_emb = self.TSModel(demogra, ts_data, ts_tt, ts_mask, ts_tau)

        query_ts_tt = query_ts_tt[:, self.contrastive_num:]
        query_ts_data = query_ts_data[:, self.contrastive_num:]
        query_ts_mask = query_ts_mask[:, self.contrastive_num:]

        ts_mtand_agg_rep = self.ts_mtand(query_ts_tt, ts_rep_tt, ts_rep, ts_emb_mask)  #
        # mfand
        ts_query_value = torch.cat([query_ts_data, query_ts_mask], dim=2)
        ts_key_value = torch.cat([ts_data, ts_mask], dim=2)
        ts_mfand_agg_rep, ts_impute_rep = self.ts_mfand(ts_query_value, ts_key_value, ts_rep, ts_emb_mask,
                                                        query_ts_data)

        mtand_mfand_moe_gate = torch.cat([ts_mtand_agg_rep, ts_mfand_agg_rep], dim=-1)
        mtand_mfand_mixup_rate = self.mtand_mfand_moe(mtand_mfand_moe_gate)
        ts_agg_rep = mtand_mfand_mixup_rate * ts_mtand_agg_rep + (1 - mtand_mfand_mixup_rate) * ts_mfand_agg_rep

        if self.mixup_level == 'batch':
            g_time = torch.max(ts_agg_rep, dim=0).values
            g_value = torch.max(ts_impute_rep, dim=0).values
            moe_gate = torch.cat([g_time, g_value], dim=-1)  # L_t+l, 2D
        elif self.mixup_level == 'batch_seq' or self.mixup_level == 'batch_seq_feature':
            moe_gate = torch.cat([ts_agg_rep, ts_impute_rep], dim=-1)
        else:
            raise ValueError("Unknown mixedup type")
        mixup_rate = self.moe(moe_gate)
        ts_agg_rep = mixup_rate * ts_agg_rep + (1 - mixup_rate) * ts_impute_rep

        ts_restore_rep = self.ts_decoder(ts_agg_rep)  # B l K
        ts_restore_rep[restore_ts_mask==0] = 0
        restore_ts_label[restore_ts_mask==0] = 0

        return ts_restore_rep, restore_ts_label




