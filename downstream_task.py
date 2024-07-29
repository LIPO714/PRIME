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


# TODO: choose downstream tasks: 24PHENO or 48IHM
# from all_exps.downstream_24PHENO_args_config import parse_args
from all_exps.downstream_48IHM_args_config import parse_args


def main():

    args = parse_args()
    print(args)
    make_save_dir(args)
    make_log_dir(args)
    make_tensorboard_dir(args)

    dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.world_size)
    print("init process group")

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
    else:
        pass

    if float(args.data_percent) > 0.3:
        classifier = ClassificationHead(args, device)
    else:
        print("*******easy classifier*******")
        classifier = EASY_ClassificationHead(args, device)

    if args.mode == "train":
        model.load_state_dict(torch.load(args.pretrain_model, map_location=device))
    else:
        checkpoint = load_full_model(args, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])

    print("Bert parameters:", sum(p.numel() for p in model.TextModel.bertrep.parameters() if p.requires_grad))  # 148659456
    print("All model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))  # 149698753

    model.cuda(gpu)
    classifier.cuda(gpu)
    classifier = nn.SyncBatchNorm.convert_sync_batchnorm(classifier)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=True
    )
    classifier = torch.nn.parallel.DistributedDataParallel(
        classifier, device_ids=[gpu], find_unused_parameters=True
    )
    model = model.module
    classifier = classifier.module
    scaler = GradScaler(enabled=args.use_mix_precision)

    # loss function
    if args.task == "48ihm":
        pred_loss_func = nn.BCEWithLogitsLoss(reduction='none').to(device)
        # pred_loss_func = nn.CrossEntropyLoss(reduction='none').to(device)
    elif args.task == "24pheno":
        pred_loss_func = nn.BCEWithLogitsLoss(reduction='none').to(device)

    if args.mode == "train":
        train_dataset, train_sampler, train_dataloader=data_perpare(args, args.task, device, mode='train')
        val_dataset, val_sampler, val_dataloader = data_perpare(args, args.task, device, mode='val')
        test_dataset, test_sampler, test_dataloader = data_perpare(args, args.task, device, mode='test')

        print("train_dataset:", len(train_dataset))
        print("val_dataset:", len(val_dataset))
        print("test_dataset:", len(test_dataset))

        # 冻结参数
        if args.freeze == "Bert+Other":
            for param in model.parameters():
                param.requires_grad = False
        elif args.freeze == "Bert":
            for param in model.TextModel.bertrep.parameters():
                param.requires_grad = False
        elif args.freeze == "part_of_Bert":
            for name, param in model.TextModel.bertrep.named_parameters():
                # freeze
                if 'encoder.layer.6' not in name and 'encoder.layer.7' not in name and 'encoder.layer.8' not in name and 'encoder.layer.9' not in name and 'encoder.layer.10' not in name and 'encoder.layer.11' not in name:
                # if 'encoder.layer.9' not in name and 'encoder.layer.10' not in name and 'encoder.layer.11' not in name:

                    param.requires_grad = False
        elif args.freeze == "None":
            pass

        if args.task == "48ihm":
            args.steps_per_epoch = len(train_dataloader)
        else:
            args.steps_per_epoch = len(train_dataloader) / args.gradient_accumulation_steps

        bert_optimizer, other_optimizer = create_optimizer(args, model)
        class_optimizer = create_class_optimizer(args, classifier)

        bert_scheduler, other_scheduler = create_scheduler(args, bert_optimizer, other_optimizer)
        class_scheduler = create_class_scheduler(args, class_optimizer)

        metadata = read_metadata(args.metadata)

        trainer_downstream(model=model,classifier=classifier,args=args,metadata=metadata,scaler=scaler,train_dataloader=train_dataloader,val_dataloader=val_dataloader, test_dataloader=test_dataloader, device=device, loss_func=pred_loss_func, \
            bert_optimizer=bert_optimizer, other_optimizer=other_optimizer, class_optimizer=class_optimizer, bert_scheduler=bert_scheduler, \
                         other_scheduler=other_scheduler, class_scheduler=class_scheduler, writer=writer)

    else:
        for param in model.parameters():
            param.requires_grad = False
        test_dataset, test_sampler, test_dataloader = data_perpare(args, args.task, device, mode='test')

        metadata = read_metadata(args.metadata)

        tester_downstream(model=model, classifier=classifier, args=args, metadata=metadata, scaler=scaler,
                           test_dataloader=test_dataloader,
                           device=device,
                           loss_func=pred_loss_func,
                           writer=writer)

    dist.destroy_process_group()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))