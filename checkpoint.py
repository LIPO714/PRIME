import re
import os
import torch
import numpy as np
import operator
from statistics import mean,stdev
import fnmatch

import shutil


def make_save_dir(args):
    save_dir = args.save_dir + args.task + "/" + args.exp_name + "_seed_" + str(args.seed) + "_percent_" + str(args.data_percent) + \
                         "_epoch_" + str(args.num_train_epochs) + "_batch_" + str(args.train_batch_size) + "_" + \
                         args.model_name + str(args.max_length) + "_bert_update_" + str(args.num_update_bert_epochs) + "_" + str(args.bertcount) + \
                         "_notenum_" + str(args.num_of_notes) + "_emb_dim_" + str(args.ts_embed_dim) + "_bert_lr_" + str(args.bert_lr) + "_other_lr_" + str(args.other_lr)
    if hasattr(args, 'class_lr'):
        save_dir += '_class_lr_' + str(args.class_lr)
    args.save_dir=save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(args.save_dir)

def make_log_dir(args):
    log_dir = args.log_dir + args.task + "/" + args.exp_name + "_seed_" + str(args.seed) + "_percent_" + str(args.data_percent) + \
                         "_epoch_" + str(args.num_train_epochs) + "_batch_" + str(args.train_batch_size) + "_" + \
                         args.model_name + str(args.max_length) + "_bert_update_" + str(args.num_update_bert_epochs) + "_" + str(args.bertcount) + \
                         "_notenum_" + str(args.num_of_notes) + "_emb_dim_" + str(args.ts_embed_dim) + "_bert_lr_" + str(args.bert_lr) + "_other_lr_" + str(args.other_lr)
    if args.class_lr is not None:
        log_dir += '_class_lr_' + str(args.class_lr)
    args.log_dir = log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

def make_tensorboard_dir(args):
    out_dir = args.tensorboard_dir + args.task + "/" + args.exp_name + "_seed_" + str(args.seed) + "_percent_" + str(args.data_percent) + \
                         "_epoch_" + str(args.num_train_epochs) + "_batch_" + str(args.train_batch_size) + "_" + \
                         args.model_name + str(args.max_length) + "_bert_update_" + str(args.num_update_bert_epochs) + "_" + str(args.bertcount) + \
                         "_notenum_" + str(args.num_of_notes) + "_emb_dim_" + str(args.ts_embed_dim) + "_bert_lr_" + str(args.bert_lr) + "_other_lr_" + str(args.other_lr)
    if hasattr(args, 'class_lr'):
        out_dir += '_class_lr_' + str(args.class_lr)
    args.tensorboard_dir = out_dir
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)


def save_pretrain_ckpt(save_path, model):
    torch.save(model.state_dict(), save_path)

def load_full_model(args, device):
    return torch.load(args.full_model, map_location=device)

def load_best_full_model(args, best_epoch, device):
    save_dir = args.save_dir
    best_ck = save_dir + '/' + str(best_epoch) + '.pth'
    return torch.load(best_ck, map_location=device)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, save_dir=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_dir = save_dir
        self.best_epoch = -1

    def __call__(self, val_loss, model, classifier=None, time_predictor=None, decoder=None, epoch=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, time_predictor, decoder, epoch)
            if epoch is not None:
                self.best_epoch = epoch
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience} ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, time_predictor, decoder, epoch)
            if epoch is not None:
                self.best_epoch = epoch
            self.counter = 0

    def save_checkpoint(self, val_loss, model, classifier=None, time_predictor=None, decoder=None, epoch=None):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        classifier_state_dict = None

        model_state_dict = model.state_dict()

        if classifier is not None:
            classifier_state_dict = classifier.state_dict()

        if self.save_dir is not None:
            save_path = self.save_dir + '/' + str(epoch) + '.pth'
            torch.save({
                'model_state_dict': model_state_dict,
                'classifier_state_dict': classifier_state_dict,
            }, save_path)
        else:
            print("no path assigned")

        self.val_loss_min = val_loss


if __name__ == "__main__":
    dst='test/'
    copy_file(dst, src=os.getcwd())
