import random
import torch
import gc
import numpy as np
import time
import wandb

from tqdm import tqdm

from agumentation.d_agumentation import demogra_agumentation_bank
from agumentation.its_agumentation import its_agumentation_bank
from agumentation.note_agumentation_subpad import note_agumentation_bank
from checkpoint import save_pretrain_ckpt, EarlyStopping, load_best_full_model
from util import evaluate_ml, evaluate_mc, log_info


categorical_col = [3, 4, 5, 6]
categorical_dict = {}
categorical_dict[0] = {
    "name": "Glascow coma scale eye opening",
    "start": -1.85508,
    "end": 0.64865597,
    "num": 4,
    "eps": 0.01,
}

categorical_dict[1] = {
    "name": "Glascow coma scale motor response",
    "start": -3.02190756,
    "end": 0.51430339,
    "num": 6,
    "eps": 0.01,
}

categorical_dict[2] = {
    "name": "Glascow coma scale total",
    "start": -2.24015702,
    "end": 0.92002567,
    "num": 13,  # 3-15
    "eps": 0.01,
}

categorical_dict[3] = {
    "name": "Glascow coma scale verbal response",
    "start": -1.07510832,
    "end": 1.02388242,
    "num": 5,
    "eps": 0.01,
}


def eval_state(model,args,metadata,scaler,train_dataloader,bert_optimizer,other_optimizer,bert_scheduler=None,other_scheduler=None,writer=None):

    for item in range(len(categorical_col)):
        start = categorical_dict[item]["start"]
        end = categorical_dict[item]["end"]
        num = categorical_dict[item]["num"] - 1
        interval = (end - start) / num
        list = []
        now = start
        while now < end - 0.0001:
            list.append(now)
            now += interval
        list.append(end)
        categorical_dict[item]["list"] = list
        categorical_dict[item]["pred_eps"] = (interval / 2) * 0.99
        print("list:", list)


    count=0
    global_step=0
    agu_choice = ['its', 'note']
    for epoch in tqdm(range(args.num_train_epochs)):
        ################    N4    ################
        train_dataloader.sampler.set_epoch(epoch)  #
        ##########################################
        model.eval()


        # for step, batch in tqdm(enumerate(train_dataloader)):
        len_batch = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            # if step < 58:
            #     continue
            # print("1....")
            global_step += 1

            # 1. preprocess data
            name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_tt, note_tau, note_mask, restore_index, restore_ts_label, restore_ts_mask, query_ts_tt, query_note_tt, query_ts_data, query_ts_mask = batch

            demogra = demogra_agumentation_bank(demogra, metadata)

            # print("3....")
            # 3. input to model
            # with torch.cuda.amp.autocast(enabled=args.use_mix_precision):
            ts_restore_rep, restore_ts_label = model.eval_state(demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_tt,
                             note_tau, note_mask, restore_index, restore_ts_label, restore_ts_mask, query_ts_tt, query_note_tt, query_ts_data, query_ts_mask=query_ts_mask)

            pred_categorical = ts_restore_rep[:, :, categorical_col].cuda().cpu().detach().numpy()
            truth_categorical = restore_ts_label[:, :, categorical_col].cuda().cpu().detach().numpy()
            restore_ts_mask = restore_ts_mask[:, :, categorical_col].cuda().cpu().detach().numpy()

            pred_categorical_to_type = np.zeros_like(pred_categorical)
            truth_categorical_to_type = np.zeros_like(truth_categorical)
            for item in range(len(categorical_col)):
                now_col = pred_categorical[:, :, item]  # B L
                new_col = np.zeros_like(now_col)
                eps = categorical_dict[item]["pred_eps"]
                for type in range(categorical_dict[item]["num"]):
                    val = categorical_dict[item]["list"][type]
                    mask = np.logical_and(now_col > val - eps, now_col < val + eps)
                    new_col[mask] = type + 1
                pred_categorical_to_type[:, :, item] = new_col

                now_col = truth_categorical[:, :, item]
                new_col = np.zeros_like(now_col)
                eps = categorical_dict[item]["eps"]
                for type in range(categorical_dict[item]["num"]):
                    val = categorical_dict[item]["list"][type]
                    mask = np.logical_and(now_col > val - eps, now_col < val + eps)
                    new_col[mask] = type + 1
                truth_categorical_to_type[:, :, item] = new_col

            pred_categorical_to_type[restore_ts_mask==0] = 0

            B = ts_restore_rep.shape[0]
            for i in range(B):
                Glascow_pred = pred_categorical_to_type[i, :, 2]
                Glascow_truth = truth_categorical_to_type[i, :, 2]

                print("Glascow_pred:", Glascow_pred)
                print("Glascow_truth:", Glascow_truth)

    if writer is not None:
        writer.close()