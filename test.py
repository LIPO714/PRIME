import random
import torch
import gc
import numpy as np
import time

from tqdm import tqdm

from agumentation.d_agumentation import demogra_agumentation_bank
from agumentation.its_agumentation import its_agumentation_bank
from agumentation.note_agumentation_subpad import note_agumentation_bank
from checkpoint import save_pretrain_ckpt, EarlyStopping
from train import trainer_downstream_eval_epoch
from util import evaluate_ml, evaluate_mc, log_info
import matplotlib.pyplot as plt
import seaborn as sns
import os


def tester_downstream(model,classifier,args,metadata,scaler,test_dataloader,device,loss_func,writer=None):
    log_path = args.log_dir + '/log.out'

    # Test epoch
    start = time.time()
    test_acc, test_auroc, test_auprc, test_f1, test_auroc_micro, test_loss, similar_dict = trainer_downstream_eval_epoch(0, model, classifier, args, metadata,
                                                                            scaler, test_dataloader, device,
                                                                            loss_func)
    similar_list = similar_dict["similar_martix_list"]
    cont_similar_list = similar_dict["cont_similar_martix_list"]

    print("sim len:", len(similar_list))

    for i, sim in enumerate(similar_list):
        # sns.set()
        # plt.figure(figsize=(10, 8))
        # ax = sns.heatmap(sim, cmap="YlGnBu", annot=False)

        fig, axs = plt.subplots(1, 2, figsize=(20, 8))

        # 使用 seaborn 绘制第一个热力图
        sns.heatmap(sim, cmap="YlGnBu", ax=axs[0])
        axs[0].set_title('Similarity Matrix 1')

        # 使用 seaborn 绘制第二个热力图
        sns.heatmap(cont_similar_list[i], cmap="YlGnBu", ax=axs[1])
        axs[1].set_title('Cont Similarity Matrix')

        # 创建保存图片的目录
        output_dir = "./pic"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存热力图图片
        output_file = os.path.join(output_dir, str(i) + ".png")
        plt.savefig(output_file)
        plt.close()
        print(str(i) + ".png saved...")

    log_info(log_path, 'Test', 0, test_acc, start=start, auroc=test_auroc, auprc=test_auprc, f1=test_f1,
             auroc_micro=test_auroc_micro, loss=test_loss, save=True)

    print(
        f'Test loss: ({test_loss:.6f}).  Test acc: ({test_acc:.6f}).  Test auroc: ({test_auroc:.6f}).  Test auprc: ({test_auprc:.6f}).  Test f1: ({test_f1:.6f}).  Test auroc(micro): ({test_auroc_micro:.6f}).')




