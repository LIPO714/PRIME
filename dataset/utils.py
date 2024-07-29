from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.eval_state_dataset import EvalStateDataset, eval_state_collate_fn
from dataset.ihmDataset import Ihm48Dataset, ihm_collate_fn
from dataset.phenoDataset import Pheno24Dataset, pheno_collate_fn
from dataset.pretrainDataset_subpad import PretrainDataset, pretrain_collate_fn


def data_perpare(args, task, device, mode='train'):
    if task == "pretrain":
        dataset=PretrainDataset(args=args, data_path=args.pretrain_data, device=device, note_length=args.max_length, note_num=args.num_of_notes, bert=args.model_name, ts_restore_len=args.ts_restore_len, contrastive_num=args.contrastive_num)
        # sampler = RandomSampler(dataset)
        # dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size,
        #                         collate_fn=pretrain_collate_fn)
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=args.train_batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=False,
                                sampler=sampler,
                                collate_fn=pretrain_collate_fn)

    elif task == "eval_state":
        dataset = EvalStateDataset(args=args, data_path=args.pretrain_data, device=device, note_length=args.max_length,
                                  note_num=args.num_of_notes, bert=args.model_name, ts_restore_len=args.ts_restore_len,
                                  contrastive_num=args.contrastive_num)
        # sampler = RandomSampler(dataset)
        # dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size,
        #                         collate_fn=pretrain_collate_fn)
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=args.train_batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=False,
                                sampler=sampler,
                                collate_fn=eval_state_collate_fn)


    elif task == "48ihm":
        if mode == 'train':
            if args.cate_dim > 0:
                data_path = args.ihm_train_data + args.data_percent + '_split_categorical.pkl'
            else:
                data_path = args.ihm_train_data + args.data_percent + '.pkl'
            dataset = Ihm48Dataset(args=args, data_path=data_path, device=device, note_length=args.max_length, note_num=args.num_of_notes, bert=args.model_name, ts_max_len=args.ts_max_len)
            # sampler = RandomSampler(dataset)
            # dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size,
            #                         collate_fn=ihm_collate_fn)
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=args.train_batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False,
                                    sampler=sampler,
                                    collate_fn=ihm_collate_fn)
        elif mode == 'val':
            if args.cate_dim > 0:
                data_path = args.ihm_val_data + args.data_percent + '_split_categorical.pkl'
            else:
                data_path = args.ihm_val_data + args.data_percent + '.pkl'
            dataset = Ihm48Dataset(args=args, data_path=data_path, device=device, note_length=args.max_length,
                                   note_num=args.num_of_notes, bert=args.model_name, ts_max_len=args.ts_max_len)
            # sampler = SequentialSampler(dataset)  # 按顺序
            # dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,
            #                         collate_fn=ihm_collate_fn)
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=args.eval_batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False,
                                    sampler=sampler,
                                    collate_fn=ihm_collate_fn)
        elif mode == 'test':
            if args.cate_dim > 0:
                data_path = args.ihm_test_data + args.data_percent + '_split_categorical.pkl'
            else:
                data_path = args.ihm_test_data + args.data_percent + '.pkl'
            dataset = Ihm48Dataset(args=args, data_path=data_path, device=device, note_length=args.max_length,
                                   note_num=args.num_of_notes, bert=args.model_name, ts_max_len=args.ts_max_len)
            # sampler = SequentialSampler(dataset)  # 按顺序
            # dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,
            #                         collate_fn=ihm_collate_fn)
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=args.eval_batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False,
                                    sampler=sampler,
                                    collate_fn=ihm_collate_fn)
    elif task == "24pheno":
        if mode == 'train':
            if args.cate_dim > 0:
                data_path = args.pheno_train_data + args.data_percent + '_split_categorical.pkl'
            else:
                data_path = args.pheno_train_data + args.data_percent + '.pkl'
            dataset = Pheno24Dataset(args=args, data_path=data_path, device=device, note_length=args.max_length,
                                   note_num=args.num_of_notes, bert=args.model_name, ts_max_len=args.ts_max_len)
            # sampler = RandomSampler(dataset)
            # dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size,
            #                         collate_fn=pheno_collate_fn)
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=args.train_batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False,
                                    sampler=sampler,
                                    collate_fn=pheno_collate_fn)
        elif mode == 'val':
            if args.cate_dim > 0:
                data_path = args.pheno_val_data + args.data_percent + '_split_categorical.pkl'
            else:
                data_path = args.pheno_val_data + args.data_percent + '.pkl'
            dataset = Pheno24Dataset(args=args, data_path=data_path, device=device, note_length=args.max_length,
                                     note_num=args.num_of_notes, bert=args.model_name, ts_max_len=args.ts_max_len)
            # sampler = SequentialSampler(dataset)  # 按顺序
            # dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,
            #                         collate_fn=pheno_collate_fn)
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=args.eval_batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False,
                                    sampler=sampler,
                                    collate_fn=pheno_collate_fn)
        elif mode == 'test':
            if args.cate_dim > 0:
                data_path = args.pheno_test_data + args.data_percent + '_split_categorical.pkl'
            else:
                data_path = args.pheno_test_data + args.data_percent + '.pkl'
            dataset = Pheno24Dataset(args=args, data_path=data_path, device=device, note_length=args.max_length,
                                     note_num=args.num_of_notes, bert=args.model_name, ts_max_len=args.ts_max_len)
            # sampler = SequentialSampler(dataset)  # 按顺序
            # dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,
            #                         collate_fn=pheno_collate_fn)
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=args.eval_batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False,
                                    sampler=sampler,
                                    collate_fn=pheno_collate_fn)

    return dataset, sampler, dataloader