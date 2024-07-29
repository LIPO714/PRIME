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
import torch.distributed as dist
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast

from dataset.utils import data_perpare
from eval import eval_state
from model.multi_model import MULTCrossModel
from optim.optim_factory import create_optimizer
from scheduler.scheduler_factory import create_scheduler
from train import trainer_pretrain

logger = logging.getLogger(__name__)
from accelerate import Accelerator
# from model import *
# from train import *
from checkpoint import *
from util import *
from all_exps.pretrain_args_config import parse_args



def main():
    # 开启异常检测
    # torch.autograd.set_detect_anomaly(True)

    args = parse_args()
    print(args)
    make_save_dir(args)
    make_tensorboard_dir(args)

    dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.world_size)
    print("init process group")

    if torch.cuda.is_available():
        # device = 'cpu'
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

    if args.seed==0:
        copy_file(args.ck_file_path+'model/', src=os.getcwd())
    args.mode = 'train'
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

    train_dataset, train_sampler, train_dataloader=data_perpare(args, args.task, device)

    args.steps_per_epoch = len(train_dataloader)

    bert_optimizer, other_optimizer = create_optimizer(args, model)

    bert_scheduler, other_scheduler = create_scheduler(args, bert_optimizer, other_optimizer)

    metadata = read_metadata(args.metadata)

    trainer_pretrain(model=model,args=args,metadata=metadata,scaler=scaler,train_dataloader=train_dataloader,\
        bert_optimizer=bert_optimizer, other_optimizer=other_optimizer, bert_scheduler=bert_scheduler, \
                     other_scheduler=other_scheduler, writer=writer)

    dist.destroy_process_group()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))