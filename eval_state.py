import time

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import warnings
import time
import logging
import random

from dataset.utils import data_perpare
from eval import eval_state
from model.classifier import ClassificationHead, EASY_ClassificationHead
from model.multi_model import MULTCrossModel
from optim.optim_factory import create_optimizer, create_class_optimizer
from scheduler.scheduler_factory import create_scheduler, create_class_scheduler
from test import tester_downstream
from train import trainer_pretrain, trainer_downstream

logger = logging.getLogger(__name__)
from accelerate import Accelerator
import torch.optim as optim
# from model import *
# from train import *
from checkpoint import *
from util import *
# from interp import *

import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

# from config.downstream_args_config import parse_args  # 原始
# from all_exps.downstream_0_args_config import parse_args  # 原始
# from all_exps.downstream_1_args_config import parse_args  # exp_1: mtand no value transfer
# from all_exps.downstream_2_args_config import parse_args  # exp_2: backward impute
# from all_exps.downstream_3_args_config import parse_args  # exp_3: bert update after 2 epoch
# from all_exps.downstream_4_args_config import parse_args  # exp_4: note encoder only val
# from all_exps.downstream_5_args_config import parse_args  # exp_5: mtand no value transfer & backward impute
# from all_exps.downstream_pheno_5_args_config import parse_args
# from all_exps.downstream_6_args_config import parse_args  # exp_6: mtand no value transfer & note encoder only val
# from all_exps.downstream_7_args_config import parse_args  # exp_7: mtand no value transfer & backward impute & gate feature
# from all_exps.downstream_8_args_config import parse_args  # exp_8: mtand no value transfer & backward impute & gate feature & cont 0.75
# from all_exps.downstream_9_args_config import parse_args  # exp_9: mtand no value transfer & backward impute & note only val
# from all_exps.downstream_10_args_config import parse_args  # exp_10: mtand no value transfer & backward impute & cate emb
# from all_exps.downstream_11_args_config import parse_args  # exp_11: mtand no value transfer & backward+linear impute & cate emb
from all_exps.downstream_12_eval_state_args_config import parse_args  # exp_12: new structure
# from all_exps.downstream_12_pheno_args_config import parse_args   # exp_12: new structure
# from all_exps.downstream_13_args_config import parse_args  # exp_13: new structure mtand mvand feature gate
# from all_exps.downstream_14_args_config import parse_args  # exp_14: new structure bert avg pooling



def main():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # 开启异常检测
    # torch.autograd.set_detect_anomaly(True)

    args = parse_args()
    print(args)
    make_save_dir(args)
    make_log_dir(args)
    make_tensorboard_dir(args)

    ########################################    N1    ####################################################################
    dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.world_size)  #
    print("init process group")
    ######################################################################################################################

    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpuid)
        gpu = args.gpuid
        # pass
    else:
        device = 'cpu'

    print("Device:", device)
    if args.tensorboard_dir!=None:
        writer = SummaryWriter(args.tensorboard_dir)
    else:
        writer=None

    warnings.filterwarnings('ignore')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.seed==0:
        copy_file(args.ck_file_path+'model/', src=os.getcwd())
    BioBert, BioBertConfig, tokenizer = loadBert(args, device)
    if 'Text' in args.modeltype:
        model = MULTCrossModel(args=args, Biobert=BioBert)
        # model= MULTCrossModel(args=args,device=device,orig_d_ts=17, orig_reg_d_ts=34, orig_d_txt=768,ts_seq_num=args.tt_max,text_seq_num=args.num_of_notes,Biobert=BioBert)
    else:
        pass
        # model= TSMixed(args=args,device=device,orig_d_ts=17,orig_reg_d_ts=34, ts_seq_num=args.tt_max)

    model.cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=True
    )
    model = model.module
    scaler = GradScaler(enabled=args.use_mix_precision)

    model.load_state_dict(torch.load(args.pretrain_model, map_location=device))

    train_dataset, train_sampler, train_dataloader = data_perpare(args, args.task, device)

    args.steps_per_epoch = len(train_dataloader)

    bert_optimizer, other_optimizer = create_optimizer(args, model)

    bert_scheduler, other_scheduler = create_scheduler(args, bert_optimizer, other_optimizer)

    metadata = read_metadata(args.metadata)

    eval_state(model=model, args=args, metadata=metadata, scaler=scaler, train_dataloader=train_dataloader, \
                     bert_optimizer=bert_optimizer, other_optimizer=other_optimizer, bert_scheduler=bert_scheduler, \
                     other_scheduler=other_scheduler, writer=writer)

    ############    N8    ###########
    dist.destroy_process_group()  #
    #################################


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))