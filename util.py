import os
import sys

import pickle
import re
import numpy as np
import json
import time
import torch
from data import *
import statistics as stat
import  argparse
import pickle
from accelerate import Accelerator
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from transformers import (AutoTokenizer,
                          AutoModel,
                          AutoConfig,
                          AdamW,
                          BertTokenizer,
                          BertModel,
                          get_scheduler,
                          set_seed,
                          BertPreTrainedModel,
                          LongformerConfig,
                          LongformerModel,
                          LongformerTokenizer,

                         )

sys.path.insert(0, '../')
sys.path.insert(0, '../TS/mimic3-benchmarks')
sys.path.insert(0, '../ClinicalNotes_TimeSeries/models')
logger = None


def loadBert(args, device):
    if args.model_name!=None:
        if args.model_name== 'BioBert':
            tokenizer = AutoTokenizer.from_pretrained("./Bio_ClinicalBERT")
            BioBert=AutoModel.from_pretrained("./Bio_ClinicalBERT")
        elif args.model_name=="bioRoberta":
            config = AutoConfig.from_pretrained("allenai/biomed_roberta_base", num_labels=args.num_labels)
            tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
            BioBert = AutoModel.from_pretrained("allenai/biomed_roberta_base")
        elif  args.model_name== "Bert":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            BioBert = BertModel.from_pretrained("bert-base-uncased")
        elif args.model_name== "bioLongformer":
            tokenizer = AutoTokenizer.from_pretrained("./Clinical-Longformer")
            BioBert= AutoModel.from_pretrained("./Clinical-Longformer")
        else:
            raise ValueError("model_name should be BioBert,bioRoberta,bioLongformer or Bert")
    else:
        if args.model_path!=None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            BioBert = AutoModel.from_pretrained(args.model_path)
        else:
            raise ValueError("provide either model_name or model_path")

    BioBertConfig = BioBert.config
    BioBert = BioBert.to(device)
    return BioBert, BioBertConfig, tokenizer


def data_generate(args):
    dataPath = os.path.join(args.file_path,  'all_data_p2x_data.pkl')
    if os.path.isfile(dataPath):
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            data = pickle.load(f)
            if args.debug:
                data=data[:100]

    data=np.array(data)
    total_num=len(data)
    idx=np.arange(total_num)

    np.random.seed(args.seed)
    np.random.shuffle(idx)

    train= data[idx[:int(len(idx)*0.8)]]
    print(train[0]['data_names'])
    val=data[idx[int(len(idx)*0.8):int(len(idx)*0.9)]]
    test=data[idx[int(len(idx)*0.9):]]

    train=train.tolist()
    val=val.tolist()
    test=test.tolist()
    return train, val, test


def metrics_multilabel(y_true, predictions, verbose=1):
    # import pdb; pdb.set_trace()
    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")

    if verbose:
        # print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return{"auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted}


def diff_float(time1, time2):
    h = (time2-time1).astype('timedelta64[m]').astype(int)
    return h/60.0

def get_time_to_end_diffs(times, starttimes):

    timetoends = []
    for times, st in zip(times, starttimes):
        difftimes = []
        et = np.datetime64(st) + np.timedelta64(49, 'h')
        for t in times:
            time = np.datetime64(t)
            dt = diff_float(time, et)
            assert dt >= 0 #delta t should be positive
            difftimes.append(dt)
        timetoends.append(difftimes)
    return timetoends

def change_data_form(file_path,mode,debug=False):
    dataPath = os.path.join(file_path, mode + '.pkl')
    if os.path.isfile(dataPath):
        # We write the processed data to a pkl file so if we did that already we do not have to pre-process again and this increases the running speed significantly
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            # (data, _, _, _) = pickle.load(f)
            data = pickle.load(f)
            if debug:
                data=data[:500]

        data_X = data[0]
        data_y = data[1]
        data_text = data[2]
        data_names = data[3]
        start_times = data[4]
        timetoends = data[5]

        dataList=[]

        assert len(data_X)==len(data_y)==len(data_text)==len(data_names)==len(start_times)==len(timetoends) 


        assert  len(data_text[0])==len(timetoends[0])
        for x,y, text, name, start, end in zip(data_X,data_y,data_text, data_names,start_times,timetoends):
            if len(text)==0:
                continue
            new_text=[]
            for t in text:
                # import pdb;
                # pdb.set_trace()
                t=re.sub(r'\s([,;?.!:%"](?:\s|$))', r'\1', t)
                t=re.sub(r"\b\s+'\b", r"'", t)
                new_text.append(t.lower().strip())


            data_detail={"data_names":name,
                         "TS_data":x,
                         "text_data":new_text,
                        "label":y,
                         "adm_time":start,
                         "text_time_to_end":end
                        }
            dataList.append(data_detail)

    os.makedirs('Data',exist_ok=True)
    dataPath2 = os.path.join(file_path, mode + 'p2x_data.pkl')

    with open(dataPath2, 'wb') as f:
        # Write the processed data to pickle file so it is faster to just read later
        pickle.dump(dataList, f)

    return dataList

def data_replace(file_path1,file_path2,mode,debug=False):
    dataPath1 = os.path.join(file_path2, mode + '.pkl')
    dataPath2 = os.path.join(file_path1, mode + 'p2x_data.pkl')
    if os.path.isfile(dataPath1):
        # We write the processed data to a pkl file so if we did that already we do not have to pre-process again and this increases the running speed significantly
        print('Using', dataPath1)
        with open(dataPath1, 'rb') as f:
            data = pickle.load(f)
            if debug:
                data=data[:500]

    with open(dataPath2, 'rb') as f:
            data_r=pickle.load(f)
    data_X = data[0]
    data_y = data[1]
    data_text = data[2]
    data_names = data[3]
    start_times = data[4]
    timetoends = data[5]
    data_dict={}

    assert len(data_X)==len(data_y)==len(data_text)==len(data_names)==len(start_times)==len(timetoends) 



    assert  len(data_text[0])==len(timetoends[0])
    for x,name in zip(data_X, data_names):

        data_dict[name]=x
    for idx, data_detail in enumerate(data_r):
        new_x=data_dict[data_detail['data_names']]
        data_detail['TS_data']=new_x


    dataPath3=os.path.join(file_path2, mode + 'p2x_data.pkl')
    with open(dataPath3, 'wb') as f:
        pickle.dump(data_r, f)





def merge_reg_irg(dataPath_reg, dataPath_irg):
    with open(dataPath_irg, 'rb') as f:
        data_irg=pickle.load(f)

    with open(dataPath_reg, 'rb') as f:
        data_reg=pickle.load(f)


    for idx, data_dict in enumerate(data_reg):
        irg_dict=data_irg[data_dict['data_names']]
        data_dict['ts_tt']=irg_dict['ts_tt']
        data_dict['irg_ts']=irg_dict['irg_ts']
        data_dict['irg_ts_mask']=irg_dict['irg_ts_mask']

        assert (data_dict['label']==irg_dict['label']).all()

    with open(dataPath_reg, 'wb') as f:
        pickle.dump(data_reg,f)


def read_metadata(path):
    with open(path, 'r') as json_file:
        meta = json.load(json_file)
        return meta


def evaluate_mc(label, pred, task):
    if task == "24pheno":
        pred_labels = np.array(pred)
        label_classes = np.array(label)

        try:
            auroc = metrics.roc_auc_score(label_classes, pred_labels, average='macro')
            auprc_macro = metrics.average_precision_score(label_classes, pred_labels, average='macro')
            auroc_micro = metrics.roc_auc_score(label_classes, pred_labels, average='micro')
        except ValueError:
            auroc = 0
            auprc_macro = 0
            auroc_micro = 0

        pred_labels = np.where(pred_labels > 0.5, 1, 0)

        acc = metrics.accuracy_score(label_classes, pred_labels)
        f1_macro = metrics.f1_score(label_classes, pred_labels, average='macro')

    return acc, auroc, auprc_macro, f1_macro, auroc_micro


def evaluate_ml(true, pred):
    # print("true:", true)
    # print("pred:", pred)
    auroc = metrics.roc_auc_score(true, pred, average='macro')
    auprc = metrics.average_precision_score(true, pred, average='macro')
    auroc_micro = metrics.roc_auc_score(true, pred, average='micro')

    preds_label = np.array(pred > 0.5, dtype=float)
    f1 = metrics.f1_score(true, preds_label)
    acc = metrics.accuracy_score(true, preds_label)

    return acc, auroc, auprc, f1, auroc_micro


def log_info(log_path, phase, epoch, acc, rmse=0.0, start=0.0, value_rmse=0.0, auroc=0.0, auprc=0.0, f1=0.0, auroc_micro=0.0, loss=0.0, save=False):
    print('  -(', phase, ') epoch: {epoch}, RMSE: {rmse: 8.5f}, acc: {type: 8.5f}, '
                'AUROC: {auroc: 8.5f}, AUPRC: {auprc: 8.5f}, F1: {f1: 8.5f}, AUROC_micro: {auroc_micro: 8.5f}, Value_RMSE: {value_rmse: 8.5f}, loss: {loss: 8.5f}, elapse: {elapse:3.3f} min'
                .format(epoch=epoch, type=acc, rmse=rmse, auroc=auroc, auprc=auprc, f1=f1, auroc_micro=auroc_micro, value_rmse=value_rmse, loss=loss, elapse=(time.time() - start) / 60))

    if save and log_path is not None:
        with open(log_path, 'a') as f:
            f.write(phase + ':\t{epoch}, TimeRMSE: {rmse: 8.5f},  ACC: {acc: 8.5f}, AUROC: {auroc: 8.5f}, AUPRC: {auprc: 8.5f}, F1: {f1: 8.5f}, AUPRC_micro: {auroc_micro}, ValueRMSE: {value_rmse: 8.5f}, Loss: {loss: 8.5f}\n'
                    .format(epoch=epoch, acc=acc, rmse=rmse, auroc=auroc, auprc=auprc, f1=f1, auroc_micro=auroc_micro, value_rmse=value_rmse, loss=loss))


def impute_ts(query_ts_tt, ts_data, ts_mask, ts_tt, sort="+"):
    # print("tsdata:", ts_data)
    # print("ts_mask:", ts_mask)
    ts_data = ts_data * ts_mask
    L, K = ts_data.shape
    L_t = query_ts_tt.shape[0]
    query_ts_data = torch.zeros((L_t, K), dtype=ts_data.dtype).to(ts_data.device)
    query_ts_mask = torch.ones((L_t, K), dtype=ts_data.dtype).to(ts_data.device) * 0.5
    query_ts_dt = torch.zeros((L_t, K), dtype=ts_data.dtype).to(ts_data.device)
    mean_data = torch.sum(ts_data, dim=0) / torch.sum(ts_mask, dim=0)
    mean_data[mean_data.isnan()] = 0

    k_sum = torch.sum(ts_mask, dim=0)
    # print("k_sum:", k_sum)
    # query_ts_mask[:, k_sum==0] = 0
    # print("query_mask:", query_ts_mask)

    if sort == "+":
        ts_data_index = 0
        query_tt_index = 0
        # 先把范围外的用均值补上，dt为0
        while query_ts_tt[query_tt_index] < ts_tt[ts_data_index]:
            query_ts_data[query_tt_index] = mean_data
            query_ts_dt[query_tt_index] = torch.zeros((K), dtype=ts_data.dtype).to(ts_data.device)
            query_tt_index += 1
            if query_tt_index >= L_t:
                break

        # 中间的 依次替换 保留上一个有效值及上一个有效值的时刻
        now_ts_data = mean_data
        now_ts_mask = torch.ones((K), dtype=ts_data.dtype).to(ts_data.device) * 0.5
        now_ts_mask[k_sum==0] = 0
        now_ts_index_data = ts_data[ts_data_index]
        now_ts_data[ts_mask[ts_data_index]==1] = now_ts_index_data[ts_mask[ts_data_index]==1]
        now_ts_mask[ts_mask[ts_data_index]==1] = 1
        now_ts_data_tt = torch.ones((K), dtype=ts_tt.dtype).to(ts_tt.device) * ts_tt[ts_data_index]
        while ts_data_index < L-1 and query_tt_index < L_t:
            if query_ts_tt[query_tt_index] >= ts_tt[ts_data_index] and query_ts_tt[query_tt_index] < ts_tt[ts_data_index + 1]:
                query_ts_data[query_tt_index] = now_ts_data
                query_ts_mask[query_tt_index] = now_ts_mask
                query_ts_dt[query_tt_index] = query_ts_tt[query_tt_index] - now_ts_data_tt
                query_tt_index += 1
                continue
            ts_data_index += 1
            now_ts_data[ts_mask[ts_data_index]==1] = ts_data[ts_data_index, ts_mask[ts_data_index]==1]
            now_ts_mask[ts_mask[ts_data_index]==1] = 1
            now_ts_data_tt[ts_mask[ts_data_index]==1] = ts_tt[ts_data_index]
        # 若超出ts tt，则继续。
        while query_tt_index < L_t and query_ts_tt[query_tt_index] > ts_tt[ts_data_index]:
            query_ts_data[query_tt_index] = now_ts_data
            query_ts_mask[query_tt_index] = now_ts_mask
            query_ts_dt[query_tt_index] = query_ts_tt[query_tt_index] - now_ts_data_tt
            query_tt_index += 1

    if sort == "-":
        ts_data_index = L-1
        query_tt_index = L_t-1
        while query_ts_tt[query_tt_index] > ts_tt[ts_data_index]:
            query_ts_data[query_tt_index] = mean_data
            query_ts_dt[query_tt_index] = torch.zeros((K), dtype=ts_data.dtype).to(ts_data.device)
            query_tt_index -= 1
            if query_tt_index < 0:
                break

        now_ts_data = mean_data
        now_ts_mask = torch.ones((K), dtype=ts_data.dtype).to(ts_data.device) * 0.5
        now_ts_mask[k_sum == 0] = 0
        now_ts_index_data = ts_data[ts_data_index]
        now_ts_data[ts_mask[ts_data_index] == 1] = now_ts_index_data[ts_mask[ts_data_index] == 1]
        now_ts_mask[ts_mask[ts_data_index] == 1] = 1
        now_ts_data_tt = torch.ones((K), dtype=ts_tt.dtype).to(ts_tt.device) * ts_tt[ts_data_index]
        while ts_data_index > 0 and query_tt_index > -1:
            if query_ts_tt[query_tt_index] <= ts_tt[ts_data_index] and query_ts_tt[query_tt_index] > ts_tt[
                ts_data_index - 1]:
                query_ts_data[query_tt_index] = now_ts_data
                query_ts_mask[query_tt_index] = now_ts_mask
                query_ts_dt[query_tt_index] = now_ts_data_tt - query_ts_tt[query_tt_index]
                query_tt_index -= 1
                continue
            ts_data_index -= 1
            now_ts_data[ts_mask[ts_data_index] == 1] = ts_data[ts_data_index, ts_mask[ts_data_index] == 1]
            now_ts_mask[ts_mask[ts_data_index] == 1] = 1
            now_ts_data_tt[ts_mask[ts_data_index] == 1] = ts_tt[ts_data_index]

        while query_tt_index > -1 and query_ts_tt[query_tt_index] < ts_tt[ts_data_index]:
            query_ts_data[query_tt_index] = now_ts_data
            query_ts_mask[query_tt_index] = now_ts_mask
            query_ts_dt[query_tt_index] = now_ts_data_tt - query_ts_tt[query_tt_index]
            query_tt_index -= 1

    return query_ts_data, query_ts_dt, query_ts_mask