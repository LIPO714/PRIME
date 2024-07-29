import pickle
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from util import impute_ts


class PretrainDataset(Dataset):
    def __init__(self, args, data_path, device, note_length, note_num, bert, ts_restore_len, contrastive_num):
        self.args = args
        self.data = self.read_pkl(data_path)
        self.need_cut = False
        self.window_len = 48
        if data_path.split('/')[-1] != "pretrain_48.pkl":
            self.need_cut = True
        self.device = device
        self.note_length = str(note_length)
        if bert == 'bioLongformer':
            self.note_index = 'note_encode_' + self.note_length + '_data'
            self.empty = torch.tensor([0, 2])
            self.pad = 1
        elif bert == 'BioBert':
            self.note_index = 'note_encode_bio_' + self.note_length + '_data'
            self.empty = torch.tensor([101, 102])
            self.pad = 0
        self.note_num = note_num
        self.bert = bert

        # categorical
        self.cate_dim = args.cate_dim
        self.continue_dim = args.var_dim - self.cate_dim
        cate_type_str = args.cate_type
        cate_type_list = cate_type_str.split("_")
        self.cate_type = []
        for i in cate_type_list:
            self.cate_type.append(int(i))

        self.conti_impute_type = args.conti_impute_type
        self.cate_impute_type = args.cate_impute_type

        # 时间序列还原的长度
        self.ts_restore_len = ts_restore_len

        # 对比学习的tt数量
        self.contrastive_num = contrastive_num

    def read_pkl(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # index = 25084  # 5967 15980
        print("getitem index:", index)
        self.now_data = self.data[index]
        name = self.now_data['name']
        demogra = self.now_data['demographics']
        demogra = torch.tensor(demogra).float().to(self.device)

        # 1. 读取数据
        # print("data 1...")
        ts_data = torch.tensor(self.now_data['ts_data']).float()
        # print("dataset ts_data.type:", ts_data.dtype)
        ts_mask = torch.tensor(self.now_data['ts_mask']).float()
        ts_tt = torch.tensor(self.now_data['ts_tt']).float()
        ts_tau = torch.tensor(self.now_data['ts_tau']).float()
        note_data = self.now_data[self.note_index]
        note_tt = self.now_data['note_tt']
        note_tau = self.now_data['note_tau']

        # 2. 选取48小时的窗口【时间序列48小时；文本72小时, 最后时刻对齐】
        # print("data 2...")
        if self.need_cut:
            ts_data, ts_tt, ts_mask, ts_tau, note_data, note_tt, note_tau = self.cut_window(ts_data, ts_tt, ts_mask, ts_tau, note_data, note_tt, note_tau)

        # print("note num:", len(note_tt))

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

        # 3. 选取5段note（随机选取）计算note的tau
        # print("data 3...")
        note_data, note_tt, note_tau, note_mask = self.choose_note(note_data, note_tt, note_tau)

        # print("---------3-----------")
        # print("note_data.shape:", len(note_data))
        # print("note_tt.shape:", note_tt.shape)
        # print("note_tt:", note_tt)
        # print("note_tau:", note_tau)
        # print("note_mask:", note_mask)

        # 4. 将note补0-->tensor
        # print("data 4...")
        note_data = pad_sequence(note_data, batch_first=True, padding_value=self.pad).to(self.device)
        note_tt = torch.tensor(note_tt).float()
        note_tau = torch.tensor(note_tau).float()

        # print("ts_data:", ts_data.shape)
        # print("note_data:", note_data.shape)
        # print("ts_tt:", ts_tt)
        # print("note_tt:", note_tt)

        note_attention_mask = torch.zeros_like(note_data)
        note_attention_mask[note_data != self.pad] = 1
        note_attention_mask = note_attention_mask.to(self.device)

        # 5. 还原片段index：随机生成哪些TS连续时刻被mask；哪个note被还原
        # print("data 5...")
        ts_len = ts_tt.shape[0]
        ts_restore_tt_start_index = random.randint(0, ts_len-self.ts_restore_len)
        ts_restore_tt_end_index = ts_restore_tt_start_index + self.ts_restore_len

        note_len = torch.sum(note_mask)
        note_restore_index = random.randint(0, note_len-1)

        restore_index = torch.tensor([ts_restore_tt_start_index, ts_restore_tt_end_index, note_restore_index], dtype=torch.int).to(self.device)  # [ts_start, ts_end, note_index]

        # 6. restore ts label: 生成还原片段的真值作为标签，现在不生成的话，后面数据增强可能改变数值
        # print("data 6...")
        restore_ts_label = ts_data[ts_restore_tt_start_index:ts_restore_tt_end_index]  # 3, K
        restore_ts_mask = ts_mask[ts_restore_tt_start_index:ts_restore_tt_end_index]  # 3, K

        # 7. query tt: cont tt + restore tt(ts tt + note tt)
        restore_ts_tt = ts_tt[ts_restore_tt_start_index:ts_restore_tt_end_index]  # 3
        restore_note_tt = note_tt[note_restore_index]  # 1
        cont_tt = self.choose_contrastive_tt(ts_tt, note_tt)  # 3

        # 8. 将ts_data, ts_tt, ts_mask, ts_tau去除restore部分
        ts_data, ts_tt, ts_mask, ts_tau = self.remove_ts_restore_part(ts_data, ts_tt, ts_mask, ts_tau, restore_index)

        # 9. imputation
        cont_tt_impute_data, cont_tt_impute_mask = self.impute_ts_data(cont_tt, ts_data, ts_mask, ts_tt)
        restore_tt_impute_data, restore_tt_impute_mask = self.impute_ts_data(restore_ts_tt, ts_data, ts_mask, ts_tt)

        query_ts_tt = torch.cat((cont_tt, restore_ts_tt), dim=0)
        restore_note_tt = torch.unsqueeze(restore_note_tt, dim=0)
        query_note_tt = torch.cat((cont_tt, restore_note_tt), dim=0)
        query_ts_data = torch.cat((cont_tt_impute_data, restore_tt_impute_data), dim=0)
        query_ts_mask = torch.cat((cont_tt_impute_mask, restore_tt_impute_mask), dim=0)

        return name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_tt, note_mask, note_tau, note_attention_mask, restore_index, restore_ts_label, restore_ts_mask, query_ts_tt, query_note_tt, query_ts_data, query_ts_mask

    def cut_window(self, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_tt, note_tau):
        now_window_len = self.window_len
        max_ts_len = self.args.max_ts_len
        tt_min = int(torch.min(ts_tt))
        tt_max = int(torch.max(ts_tt)) + 1
        if tt_max - tt_min <= self.window_len:
            start_index = 0
            end_index = ts_tt.shape[0]
            ts_start_time = float(ts_tt[0])
            if end_index - start_index > max_ts_len:  # 若时序数据太长，则随机选出150个
                select_index = self.random_select_index(start_index, end_index, max_ts_len)

                ts_data = ts_data[select_index, :]
                ts_mask = ts_mask[select_index, :]
                ts_tt = ts_tt[select_index] - ts_start_time  # [0, 48]
                ts_tau = ts_tau[select_index, :]

            del tt_max, tt_min, max_ts_len
            return ts_data, ts_tt, ts_mask, ts_tau, note_data, note_tt, note_tau

        note_start_time_expand = 24
        note_end_time_expand = 0
        note_expand_time = 12
        window_expand_time = 6
        # print("note_tt:", note_tt)
        # print("ts_tt:", ts_tt)
        times = 0
        while True:
            times += 1
            # print("now window len:", now_window_len)
            if times >= 500:
                break
            end_time = random.randint(tt_min + now_window_len, tt_max)
            ts_start_time = end_time - now_window_len
            note_start_time = ts_start_time - note_start_time_expand
            note_end_time = end_time + note_end_time_expand
            # print("ts start time:", ts_start_time)
            # print("ts end time:", end_time)
            # print("note start time:", note_start_time)
            # print("note end time:", note_end_time)
            # print("now window len:", now_window_len)

            start_note_index = np.sum(note_tt < note_start_time)
            end_note_index = np.sum(note_tt <= note_end_time)
            if end_note_index - start_note_index < 3:
                # 如果window内note数量太少，要重新选
                note_start_time_expand += (note_expand_time / 2)
                note_end_time_expand += note_expand_time
                # print("note num too less...")
                continue

            # ts:
            start_ts_index = torch.sum(ts_tt < ts_start_time)
            end_ts_index = torch.sum(ts_tt <= end_time)
            if end_ts_index - start_ts_index <= 2 * self.ts_restore_len:  # 若小于2倍需要填补的，则需要重新选择
                now_window_len += window_expand_time
                if tt_min + now_window_len > tt_max:
                    now_window_len = tt_max - tt_min
                continue
            if end_ts_index - start_ts_index > max_ts_len:  # 若时序数据太长，则随机选出150个
                select_index = self.random_select_index(start_ts_index, end_ts_index, max_ts_len)

                ts_data = ts_data[select_index, :]
                ts_mask = ts_mask[select_index, :]
                ts_tt = ts_tt[select_index] - ts_start_time  # [0, 48]
                ts_tau = ts_tau[select_index, :]
            else:
                ts_data = ts_data[start_ts_index:end_ts_index, :]
                ts_mask = ts_mask[start_ts_index:end_ts_index, :]
                ts_tt = ts_tt[start_ts_index:end_ts_index] - ts_start_time  # [0, 48]
                ts_tau = ts_tau[start_ts_index:end_ts_index, :]

            # note
            note_tt = note_tt[start_note_index:end_note_index] - ts_start_time
            note_data = note_data[start_note_index:end_note_index]
            note_tau = note_tau[start_note_index:end_note_index]

            del note_start_time_expand, note_end_time_expand, note_expand_time, max_ts_len, now_window_len
            del start_ts_index, end_ts_index, start_note_index, end_note_index, tt_max, tt_min, end_time, ts_start_time, note_start_time
            return ts_data, ts_tt, ts_mask, ts_tau, note_data, note_tt, note_tau

    def random_select_index(self, start_index, end_index, num_to_select):
        # 生成所有可能的索引列表
        all_indexes = list(range(start_index, end_index))

        # 从所有索引中随机选择num_to_select个索引
        selected_indexes = random.sample(all_indexes, num_to_select)

        # 对选择的索引进行排序
        selected_indexes.sort()

        return selected_indexes

    def choose_note(self, note_data, note_tt, note_tau):
        # 随机连续选取5段；若不够5段，则需要补充
        length = self.note_num
        note_len = len(note_data)
        note_mask = torch.ones(length).float()
        if note_len < length:
            padding_num = length - note_len
            empty_note = [self.empty for _ in range(padding_num)]
            note_data = note_data + empty_note
            note_tt = np.concatenate((note_tt, [0]*padding_num))
            note_tau = np.concatenate((note_tau, [0]*padding_num))
            note_mask[-padding_num:] = 0
        elif note_len > length:
            min = 0
            max = note_len - length
            start_index = random.randint(min, max)
            end_index = start_index + length

            note_data = note_data[start_index:end_index]
            note_tt = note_tt[start_index:end_index]
            note_tau = note_tau[start_index:end_index]

        return note_data, note_tt, note_tau, note_mask

    def choose_contrastive_tt(self, ts_tt, note_tt):
        ts_min = torch.min(ts_tt)
        ts_max = torch.max(ts_tt)
        note_min = torch.min(note_tt)
        note_max = torch.max(note_tt)

        if ts_min < note_min and ts_max > note_max:
            min = note_min
            max = note_max
        else:
            min = ts_min
            max = ts_max
        cont_tt = (max - min) * torch.rand((self.contrastive_num), dtype=ts_tt.dtype).to(ts_tt.device) + min
        cont_tt, _ = torch.sort(cont_tt)
        return cont_tt

    def impute_ts_data(self, query_ts_tt, ts_data, ts_mask, ts_tt):
        # 1. 拆成离散型和连续型的
        continue_data = ts_data[:, :self.continue_dim]

        continue_data_mask = ts_mask[:, :self.continue_dim]

        # 2. 分别进行插值
        # continue date
        if self.conti_impute_type == "linear":
            query_continue_data_plus, query_ts_dt_plus, query_continue_mask_plus = impute_ts(query_ts_tt, continue_data, continue_data_mask, ts_tt, sort="+")
            query_continue_data_minus, query_ts_dt_minus, query_continue_mask_minus = impute_ts(query_ts_tt, continue_data, continue_data_mask, ts_tt, sort="-")

            sum = query_ts_dt_minus + query_ts_dt_plus
            sum[sum == 0] = 1

            query_continue_data = query_continue_data_plus + (query_continue_data_minus - query_continue_data_plus) * query_ts_dt_plus / (sum)
            query_continue_mask = query_continue_mask_plus

        elif self.conti_impute_type == "backward":
            query_continue_data, _, query_continue_mask = impute_ts(query_ts_tt, continue_data, continue_data_mask, ts_tt, sort="+")
        elif self.conti_impute_type == "forward":
            query_continue_data, _, query_continue_mask = impute_ts(query_ts_tt, continue_data, continue_data_mask, ts_tt, sort="-")

        # cate data
        if self.cate_dim > 0:
            cate_data = ts_data[:, self.continue_dim:]
            cate_data_mask = ts_mask[:, self.continue_dim:]
            if self.cate_impute_type == "linear":
                query_cate_data_plus, query_ts_dt_plus, query_cate_mask_plus = impute_ts(query_ts_tt, cate_data, cate_data_mask, ts_tt, sort="+")
                query_cate_data_minus, query_ts_dt_minus, query_cate_mask_minus = impute_ts(query_ts_tt, cate_data, cate_data_mask, ts_tt, sort="-")

                mask = query_ts_dt_plus > query_ts_dt_minus

                query_cate_data = query_cate_data_plus
                query_cate_data[mask] = query_cate_data_minus[mask]
                query_cate_mask = query_cate_mask_plus
                query_cate_mask[query_cate_mask_minus > query_cate_mask_plus] = query_cate_data_minus[query_cate_mask_minus > query_cate_mask_plus]

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

    def remove_ts_restore_part(self, ts_data, ts_tt, ts_mask, ts_tau, restore_index):
        L, K = ts_data.shape
        l = self.ts_restore_len
        start = restore_index[0]
        end = restore_index[1]
        if start == 0:
            new_ts_data = ts_data[end:]
            new_ts_mask = ts_mask[end:]
            new_ts_tt = ts_tt[end:]
            new_ts_tau = ts_tau[end:]
        elif end == L:
            new_ts_data = ts_data[:start]
            new_ts_mask = ts_mask[:start]
            new_ts_tt = ts_tt[:start]
            new_ts_tau = ts_tau[:start]
        else:
            new_ts_data = torch.cat((ts_data[:start], ts_data[end:]), dim=0)
            new_ts_mask = torch.cat((ts_mask[:start], ts_mask[end:]), dim=0)
            new_ts_tt = torch.cat((ts_tt[:start], ts_tt[end:]), dim=0)
            new_ts_tau = torch.cat((ts_tau[:start], ts_tau[end:]), dim=0)

        ts_data = new_ts_data
        ts_mask = new_ts_mask
        ts_tt = new_ts_tt
        ts_tau = new_ts_tau

        return ts_data, ts_tt, ts_mask, ts_tau



def pretrain_collate_fn(batch):
    names, demogras, ts_datas, ts_tts, ts_masks, ts_taus, note_datas, note_tts, note_masks, note_taus, note_attention_masks, restore_indexs, restore_ts_labels, restore_ts_masks, query_ts_tt, query_note_tt, query_ts_data, query_ts_mask = zip(*batch)

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
    # print("note_datas.shape:", len(note_datas))

    note_attention_masks = list(note_attention_masks)
    # print("note_datas_attention_mask.shape:", note_datas_attention_mask)

    # print("note token type:", note_token_type.shape)

    note_tts = torch.stack(note_tts).to(device)
    note_taus = torch.stack(note_taus).to(device)
    note_masks = torch.stack(note_masks).to(device)

    restore_indexs = torch.stack(restore_indexs).to(device)
    restore_ts_labels = torch.stack(restore_ts_labels).to(device)
    restore_ts_masks = torch.stack(restore_ts_masks).to(device)

    query_ts_tt = torch.stack(query_ts_tt).to(device)
    query_note_tt = torch.stack(query_note_tt).to(device)
    query_ts_data = torch.stack(query_ts_data).to(device)
    query_ts_mask = torch.stack(query_ts_mask).to(device)

    # 1. note转成batch
    # 2. 对齐ts数据
    # 3. 转成tensor.to（device）

    return names, demogras, ts_datas, ts_tts, ts_masks, ts_taus, note_datas, note_attention_masks, note_tts, note_taus, note_masks, restore_indexs, restore_ts_labels, restore_ts_masks, query_ts_tt, query_note_tt, query_ts_data, query_ts_mask


if __name__ == '__main__':

    from all_exps.pretrain_12_args_config import parse_args

    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_value = 68  # 可以替换为任何你想要的整数种子值
    random.seed(seed_value)
    ts_restore_len = 3
    contrastive_num = 3

    # 创建CustomDataset实例
    dataset = PretrainDataset(args, data_path='../data/pretrain.pkl', device=device, note_length=512, note_num=5, bert="bioLongformer", ts_restore_len=ts_restore_len, contrastive_num=contrastive_num)

    # 创建DataLoader实例，并使用自定义的collate_fn函数
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=pretrain_collate_fn)

    # 使用DataLoader迭代数据
    for index, batch in enumerate(dataloader):
        name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_tt, note_tau, note_mask, restore_index, restore_ts_label, restore_ts_mask, query_ts_tt, query_note_tt, query_ts_data, query_ts_mask = batch
        if torch.isnan(query_ts_data).any().item():
            print("Wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(index)
            break
        print(index)

        # print(query_ts_data[0])
        # print(query_ts_mask[0])
        print("note attention mask:", note_attention_mask[0])
        # print(len(note_data))
        # print("ts_data:", ts_data.shape)
        # for note_block in note_data:
        #     print(note_block.shape)
        # 在这里添加训练代码
        # print("--------------------START---------------------------")
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
