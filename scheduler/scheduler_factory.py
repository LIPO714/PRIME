import torch
from transformers import get_linear_schedule_with_warmup

from scheduler.cosine_lr import CosineLRScheduler


def scheduler_bank(args, sche, optimizer, step_per_epoch, total_steps):
    if sche == "linear_with_warmup":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warm_up * total_steps),
                                                    num_training_steps=total_steps)
    elif sche == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_per_epoch, gamma=0.8)
    elif sche == "None":
        scheduler = None
    elif sche == "cosine":
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=args.num_train_epochs,
            t_mul=getattr(args, "lr_cycle_mul", 1.0),
            lr_min=getattr(args, "min_lr", 0.00001),
            decay_rate=getattr(args, "decay_rate", 0.8),  # 衰减比例
            warmup_lr_init=getattr(args, "warmup_lr", 0.00001),
            warmup_t=getattr(args, "warmup_epoch", 2),
            cycle_limit=getattr(args, "lr_cycle_limit", 1),
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=getattr(args, "lr_noise_pct", 0.67),
            noise_std=getattr(args, "lr_noise_std", 1.0),
            noise_seed=getattr(args, "seed", 42),
        )
    else:
        scheduler = None
        print("sche error...")
    return scheduler


def create_scheduler(args, bert_optimizer, other_optimizer):
    bert_sche = args.bert_sche
    other_sche = args.other_sche

    steps_per_epoch = args.steps_per_epoch
    total_steps = steps_per_epoch * args.num_train_epochs

    bert_scheduler = scheduler_bank(args, bert_sche, bert_optimizer, steps_per_epoch, total_steps)
    other_scheduler = scheduler_bank(args, other_sche, other_optimizer, steps_per_epoch, total_steps)

    return bert_scheduler, other_scheduler

def create_class_scheduler(args, class_optimizer):
    class_sche = args.class_sche

    steps_per_epoch = args.steps_per_epoch
    total_steps = steps_per_epoch * args.num_train_epochs

    return scheduler_bank(args, class_sche, class_optimizer, steps_per_epoch, total_steps)