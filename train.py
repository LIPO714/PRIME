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


class bcolors:
    OKGREEN = '\033[92m'  # 绿色
    FAIL = '\033[91m'      # 红色
    ENDC = '\033[0m'       # 结束颜色


# model=model,args=args,accelerator=accelerator,train_dataloader=train_dataloader, device=device, \
#         bert_optimizer=bert_optimizer, other_optimizer=other_optimizer, bert_scheduler=bert_scheduler, \
#                      other_scheduler=other_scheduler, writer=writer
def trainer_pretrain(model,args,metadata,scaler,train_dataloader,bert_optimizer,other_optimizer,bert_scheduler=None,other_scheduler=None,writer=None):
    count=0
    global_step=0
    agu_choice = ['its', 'note']
    for epoch in tqdm(range(args.num_train_epochs)):
        ################    N4    ################
        train_dataloader.sampler.set_epoch(epoch)  #
        ##########################################
        model.train()
        if "Text" in args.modeltype:
            if epoch >= args.before_update_bert_epochs and args.num_update_bert_epochs<args.num_train_epochs and (epoch) % args.num_update_bert_epochs==0 and count<args.bertcount:
                count += 1
                print("bert update at epoch " + str(epoch))
                for param in model.TextModel.bertrep.parameters():
                    param.requires_grad = True
            else:
                for param in model.TextModel.bertrep.parameters():
                    param.requires_grad = False

            for param in model.TextModel.bertrep.parameters():
                # print(epoch,param.requires_grad)
                break

        # for step, batch in tqdm(enumerate(train_dataloader)):
        len_batch = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            # if step < 58:
            #     continue
            # print("1....")
            global_step += 1

            # 1. preprocess data
            name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_tt, note_tau, note_mask, restore_index, restore_ts_label, restore_ts_mask, query_ts_tt, query_note_tt, query_ts_data, query_ts_mask = batch

            # for i in range(len(note_data)):
            #     print(note_data[i].shape)

            # print("2....")
            # 2. data agu
            agu = random.choice(agu_choice)
            if agu == 'its':
                ts_data, ts_mask, ts_tau = its_agumentation_bank(ts_data, ts_tt, ts_mask, ts_tau, metadata)
            else:  # 'note'
                if args.model_name == "bioLongformer":
                    max_token_id = metadata['max_long_token_id']
                    min_token_id = 3
                elif args.model_name == "BioBert":
                    max_token_id = metadata['max_bio_token_id']
                    min_token_id = 103
                note_data = note_agumentation_bank(note_data, args.model_name, max_token_id, min_token_id)

            demogra = demogra_agumentation_bank(demogra, metadata)

            # print("3....")
            # 3. input to model
            # with torch.cuda.amp.autocast(enabled=args.use_mix_precision):
            loss, out = model(demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_tt,
                             note_tau, note_mask, restore_index, restore_ts_label, restore_ts_mask, query_ts_tt, query_note_tt, query_ts_data, query_ts_mask=query_ts_mask)

            # print("4....")
            loss = loss / args.gradient_accumulation_steps
            # print("loss:", loss)
            # scaler.scale(loss).backward()
            loss.backward()

            if torch.isnan(loss).any().item():
                # print grad check # use for check nan
                v_n = []
                v_v = []
                v_g = []
                for name, parameter in model.named_parameters():
                    v_n.append(name)
                    v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
                    v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
                for i in range(len(v_n)):
                    if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
                        color = bcolors.FAIL + '*'
                    else:
                        color = bcolors.OKGREEN + ' '
                    print('%svalue %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
                    print('%sgrad  %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))

            # if step == 0:
            #     break

            # print("5....")

            if (step+1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # scaler.step(bert_optimizer)
                # scaler.step(other_optimizer)
                # scaler.update()
                bert_optimizer.step()
                other_optimizer.step()
                if bert_scheduler != None:
                    bert_scheduler.step()
                if other_scheduler != None:
                    other_scheduler.step()
                model.zero_grad()

            ################    N7    ####################
            if args.rank == 0:
                ##############################################
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_train_epochs, step + 1, len_batch,
                                                                         loss.item()))
                # write to tensorboard
                if writer is not None:
                    writer.add_scalar("loss", loss.item(), global_step)
                    writer.add_scalar("contrastive_loss", out['contrastive_loss'].item(), global_step)
                    writer.add_scalar("ts_restore_mse", out['ts_restore_mse'].item(), global_step)
                    writer.add_scalar("note_restore_cos", out['note_restore_cos'].item(), global_step)


            # if step == 20:
            #     break

        if args.rank == 0:
            print("saving...")
            # save model each epoch
            save_filename = args.save_dir + '/' + str(epoch) + '.pth'
            save_pretrain_ckpt(save_filename, model)

    if writer is not None:
        writer.close()


def trainer_downstream(model,classifier,args,metadata,scaler,train_dataloader,val_dataloader,test_dataloader,device,loss_func,bert_optimizer,other_optimizer,class_optimizer,bert_scheduler=None,other_scheduler=None,class_scheduler=None,writer=None):

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_dir=args.save_dir)
    log_path = args.log_dir + '/log.out'
    count=0
    global_step=0
    for epoch in range(args.num_train_epochs):

        train_dataloader.sampler.set_epoch(epoch)

        # @1 Train epoch
        print(f"Epoch {epoch}:")
        start = time.time()
        train_acc, train_auroc, train_auprc, train_f1, train_auroc_micro, train_loss, count, global_step = trainer_downstream_train_epoch(epoch, count, global_step, model, classifier,args,metadata,scaler,train_dataloader,device,loss_func,bert_optimizer,other_optimizer,class_optimizer,bert_scheduler,other_scheduler,class_scheduler,writer)
        log_info(log_path, 'Train', epoch, train_acc, start=start, auroc=train_auroc, auprc=train_auprc, f1=train_f1, auroc_micro=train_auroc_micro, loss=train_loss, save=True)

        print(f"Val...")
        start = time.time()
        val_acc, val_auroc, val_auprc, val_f1, val_auroc_micro, val_loss, _ = trainer_downstream_eval_epoch(epoch, model, classifier,args,metadata,scaler,val_dataloader,device,loss_func)
        log_info(log_path, 'Valid', epoch, val_acc, start=start, auroc=val_auroc, auprc=val_auprc, f1=val_f1, auroc_micro=val_auroc_micro, loss=val_loss, save=True)

        # @3 Test epoch
        print(f"Test...")
        start = time.time()
        test_acc, test_auroc, test_auprc, test_f1, test_auroc_micro, test_loss, _ = trainer_downstream_eval_epoch(epoch, model, classifier, args,
                                                                                    metadata, scaler, test_dataloader,
                                                                                    device, loss_func)
        log_info(log_path, 'Test', epoch, test_acc, start=start, auroc=test_auroc, auprc=test_auprc, f1=test_f1, auroc_micro=test_auroc_micro, loss=test_loss, save=True)

        if early_stopping is not None:
            early_stopping(-val_auroc, model, classifier, epoch=epoch)
            if early_stopping.early_stop:  # and not opt.pretrain:
                print("Early stopping. Training Done.")
                break


    if writer is not None:
        writer.close()

    # print("Test...")
    # best_epoch = early_stopping.best_epoch
    #
    # checkpoint = load_best_full_model(args, best_epoch, device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # classifier.load_state_dict(checkpoint['classifier_state_dict'])
    #
    # # Test epoch
    # start = time.time()
    # test_acc, test_auroc, test_auprc, test_f1, test_auroc_micro, test_loss, _ = trainer_downstream_eval_epoch(best_epoch, model, classifier, args, metadata, scaler, test_dataloader, device, loss_func)
    # log_info(log_path, 'Test', best_epoch, test_acc, start=start, auroc=test_auroc, auprc=test_auprc, f1=test_f1, auroc_micro=test_auroc_micro, loss=test_loss, save=True)
    #
    # print(f'Test loss: ({test_loss:.6f}).  Test acc: ({test_acc:.6f}).  Test auroc: ({test_auroc:.6f}).  Test auprc: ({test_auprc:.6f}).  Test f1: ({test_f1:.6f}).  Test auroc(micro): ({test_auroc_micro:.6f}).')




def trainer_downstream_train_epoch(epoch, count, global_step, model,classifier,args,metadata,scaler,train_dataloader,device,loss_func,bert_optimizer,other_optimizer,class_optimizer,bert_scheduler=None,other_scheduler=None,class_scheduler=None,writer=None):
    model.train()
    classifier.train()

    losses = []
    sup_preds, sup_labels = [], []
    acc, auroc, auprc = 0, 0, 0
    agu_choice = ['its', 'note']

    for step, batch in enumerate(tqdm(train_dataloader)):
        global_step += 1

        # 1. preprocess data
        name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask, query_tt, query_ts_data, query_ts_mask, label = batch

        # 2. input to model
        ts_rep, note_rep = model(demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_tt,
                                 note_tau, note_mask, query_ts_tt=query_tt, query_note_tt=query_tt, query_ts_data=query_ts_data, query_ts_mask=query_ts_mask)
        if torch.isnan(ts_rep).any().item():
            print("ts_rep has nan!!")
        # 3. input to classifier
        out = classifier(ts_rep, note_rep)  # B C

        # 4. loss
        if args.task == "48ihm":
            label = label.unsqueeze(1)
            loss = torch.sum(loss_func(out, label))
            sup_pred = torch.sigmoid(out)
        elif args.task == "24pheno":
            loss = torch.sum(loss_func(out, label))
            sup_pred = torch.sigmoid(out)
        else:
            loss = None
            sup_pred = None

        sup_preds.append(sup_pred.detach().cpu().numpy())
        sup_labels.append(label.detach().cpu().numpy())
        losses.append(loss.item())

        loss = loss / args.gradient_accumulation_steps

        # scaler.scale(loss).backward()
        loss.backward()

        if torch.isnan(loss).any().item():
            # print grad check # use for check nan
            v_n = []
            v_v = []
            v_g = []
            for name, parameter in model.named_parameters():
                v_n.append(name)
                v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
                v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
            for i in range(len(v_n)):
                if np.max(v_v[i]).item() - np.min(v_v[i]).item() < 1e-6:
                    color = bcolors.FAIL + '*'
                else:
                    color = bcolors.OKGREEN + ' '
                print('%svalue %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
                print('%sgrad  %s: %.3e ~ %.3e' % (color, v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))

        if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            class_optimizer.step()
            if class_scheduler != None:
                class_scheduler.step()

            if args.freeze == "Bert" or args.freeze == "part_of_Bert" or args.freeze == "None":
                other_optimizer.step()
                if other_scheduler != None:
                    other_scheduler.step()
            if args.freeze == "Other" or args.freeze == "part_of_Bert" or args.freeze == "None":
                bert_optimizer.step()
                if bert_scheduler != None:
                    bert_scheduler.step()

            model.zero_grad()
            classifier.zero_grad()

        if args.rank == 0:
            # write to tensorboard
            if writer is not None:
                writer.add_scalar("loss", loss.item(), global_step)

        del out, loss, sup_pred, label
        del ts_rep, note_rep
        del name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask, query_tt, query_ts_data

        gc.collect()
        torch.cuda.empty_cache()

    train_loss = np.average(losses)

    if len(sup_preds) > 0:
        sup_labels = np.concatenate(sup_labels)
        sup_preds = np.concatenate(sup_preds)
        sup_preds = np.nan_to_num(sup_preds)

        if args.task == '48ihm':
            acc, auroc, auprc, f1, auroc_micro = evaluate_ml(sup_labels, sup_preds)
        elif args.task == '24pheno':
            # n_classes = 25
            acc, auroc, auprc, f1, auroc_micro = evaluate_mc(sup_labels, sup_preds, args.task)

    # finish epoch
    return acc, auroc, auprc, f1, auroc_micro, train_loss, count, global_step


def trainer_downstream_eval_epoch(epoch, model, classifier, args,metadata,scaler,test_dataloader,device,loss_func):
    model.eval()
    classifier.eval()

    losses = []
    sup_preds, sup_labels = [], []
    acc, auroc, auprc = 0, 0, 0
    similar_martix_list = []
    cont_similar_martix_list = []

    for step, batch in enumerate(tqdm(test_dataloader)):

        # 1. preprocess data
        name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask, query_tt, query_ts_data, query_ts_mask, label = batch

        # 2. input to model
        ts_rep, note_rep = model(demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_tt,
                                 note_tau, note_mask, query_ts_tt=query_tt, query_note_tt=query_tt, query_ts_data=query_ts_data, query_ts_mask=query_ts_mask)

        # 3. input to classifier
        out = classifier(ts_rep, note_rep)  # B C

        # 4. loss
        if args.task == "48ihm":
            label = label.unsqueeze(1)
            loss = torch.sum(loss_func(out, label))
            sup_pred = torch.sigmoid(out)
        elif args.task == "24pheno":
            loss = torch.sum(loss_func(out, label))
            sup_pred = torch.sigmoid(out)
        else:
            loss = None
            sup_pred = None

        sup_preds.append(sup_pred.detach().cpu().numpy())
        sup_labels.append(label.detach().cpu().numpy())
        losses.append(loss.item())

        if args.mode == "eval":
            B, L, D = ts_rep.shape
            ts_rep = ts_rep.reshape(B*L, D)
            note_rep = note_rep.reshape(B*L, D)
            ts_rep_a = ts_rep[:, :64]
            note_rep_b = note_rep[:, :64]
            sim = torch.einsum("i d, j d -> i j", ts_rep, note_rep)
            cont_sim = torch.einsum("i d, j d -> i j", ts_rep_a, note_rep_b)
            sim = sim.cpu().numpy()
            cont_sim = cont_sim.cpu().numpy()
            similar_martix_list.append(sim)
            cont_similar_martix_list.append(cont_sim)
            del ts_rep_a, note_rep_b, sim, cont_sim

        del out, loss, sup_pred, label
        del ts_rep, note_rep
        del name, demogra, ts_data, ts_tt, ts_mask, ts_tau, note_data, note_attention_mask, note_token_type, note_tt, note_tau, note_mask, query_tt, query_ts_data

        gc.collect()
        torch.cuda.empty_cache()


    sim_list = {
        "similar_martix_list": similar_martix_list,
        "cont_similar_martix_list": cont_similar_martix_list,
    }

    # metric
    train_loss = np.average(losses)

    if len(sup_preds) > 0:
        sup_labels = np.concatenate(sup_labels)
        sup_preds = np.concatenate(sup_preds)
        sup_preds = np.nan_to_num(sup_preds)

        if args.task == '48ihm':
            acc, auroc, auprc, f1, auroc_micro = evaluate_ml(sup_labels, sup_preds)
        elif args.task == '24pheno':
            n_classes = 25
            acc, auroc, auprc, f1, auroc_micro = evaluate_mc(sup_labels, sup_preds, args.task)

    # finish epoch
    return acc, auroc, auprc, f1, auroc_micro, train_loss, sim_list

