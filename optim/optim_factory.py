""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import optim as optim


def optim_bank(args, opt, optimizer_parameters, lr):
    if opt == "sgd" or opt == "nesterov":
        args.pop("eps", None)
        optimizer = optim.SGD(
            optimizer_parameters, momentum=args.momentum, nesterov=True, lr=lr
        )
    elif opt == "momentum":
        args.pop("eps", None)
        optimizer = optim.SGD(
            optimizer_parameters, momentum=args.momentum, nesterov=False, lr=lr
        )
    elif opt == "adam":
        optimizer = optim.Adam(optimizer_parameters, lr=lr)
    elif opt == "adamw":
        optimizer = optim.AdamW(optimizer_parameters, lr=lr)
    elif opt == "adadelta":
        optimizer = optim.Adadelta(optimizer_parameters, lr=lr)
    elif opt == "rmsprop":
        optimizer = optim.RMSprop(
            optimizer_parameters, alpha=0.9, momentum=args.momentum, lr=lr
        )
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer


def create_optimizer(args, model):
    bert_opt = args.bert_opt.lower()
    other_opt = args.other_opt.lower()
    weight_decay = args.weight_decay

    # split bert part and other param; add weight decay
    bert_optimizer_weight_dc = []
    bert_optimizer_no_weight_dc = []
    new_optimizer_weight_dc = []
    new_optimizer_no_weight_dc = []

    no_decay = ["bias", "LayerNorm.weight"]
    skip = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    for n, p in model.named_parameters():  # name param
        if 'bert' in n:  # 如果是bert的
            if any(nd in n for nd in no_decay) or len(p.shape) == 1 or n.endswith(".bias") or n in skip:
                bert_optimizer_no_weight_dc.append(p)
            else:
                bert_optimizer_weight_dc.append(p)
        else:
            if any(nd in n for nd in no_decay) or len(p.shape) == 1 or n.endswith(".bias") or n in skip:
                new_optimizer_no_weight_dc.append(p)
            else:
                new_optimizer_weight_dc.append(p)

    bert_optimizer_parameters = [
        {
            "params": bert_optimizer_weight_dc,
            "weight_decay": weight_decay,
        },
        {
            "params": bert_optimizer_no_weight_dc,
            "weight_decay": 0.0,
        },
    ]

    new_optimizer_parameters = [
        {
            "params": new_optimizer_weight_dc,
            "weight_decay": weight_decay,
        },
        {
            "params": new_optimizer_no_weight_dc,
            "weight_decay": 0.0,
        },
    ]

    bert_optimizer = optim_bank(args, bert_opt, bert_optimizer_parameters, args.bert_lr)
    other_optimizer = optim_bank(args, other_opt, new_optimizer_parameters, args.other_lr)

    return bert_optimizer, other_optimizer


def create_class_optimizer(args, classifier):
    class_opt = args.class_opt.lower()

    return optim_bank(args, class_opt, classifier.parameters(), args.class_lr)

