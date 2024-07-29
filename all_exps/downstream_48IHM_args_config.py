import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="argparse")

    parser.add_argument("--exp_name", type=str, default="48ihm-test")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--task", type=str, default='48ihm')
    parser.add_argument("--mode", type=str, default='train', choices=["train", "eval"])
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--tensorboard_dir", type=str, default='./tensorboard/')
    parser.add_argument("--save_dir", type=str, default='./save/')
    parser.add_argument("--log_dir", type=str, default='./log/')

    # 多卡训练
    parser.add_argument('--init_method', default='tcp://localhost:18888', help="init-method")
    parser.add_argument('-g', '--gpuid', default=0, type=int, help="which gpu to use")
    parser.add_argument('-r', '--rank', default=0, type=int, help='rank of current process')
    parser.add_argument('--world_size', default=2, type=int, help="world size")
    parser.add_argument('--use_mix_precision', default=False, action='store_true', help="whether to use mix precision")

    # data
    parser.add_argument("--var_dim", type=int, default=17)
    parser.add_argument("--invar_dim", type=int, default=2)
    parser.add_argument("--cate_dim", type=int, default=0)
    parser.add_argument("--cate_type", type=str, default='4_6_13_5')
    parser.add_argument("--metadata", type=str, default='./data/metadata.json')
    parser.add_argument("--pretrain_data", type=str, default='./data/pretrain.pkl')

    parser.add_argument("--ihm_train_data", type=str, default='./data/48ihm/train_48ihm_')
    parser.add_argument("--ihm_val_data", type=str, default='./data/48ihm/val_48ihm_')
    parser.add_argument("--ihm_test_data", type=str, default='./data/48ihm/test_48ihm_')

    parser.add_argument("--pheno_train_data", type=str, default='./data/24pheno/train_24pheno_')
    parser.add_argument("--pheno_val_data", type=str, default='./data/24pheno/val_24pheno_')
    parser.add_argument("--pheno_test_data", type=str, default='./data/24pheno/test_24pheno_')

    parser.add_argument("--data_percent", type=str, default='1.0')

    # ts data
    parser.add_argument("--ts_max_len", type=int, default=220)  # 220
    parser.add_argument("--impute", default=True)
    parser.add_argument("--conti_impute_type", type=str, default="backward", choices=["linear", "backward",
                                                                                    "forward"])
    parser.add_argument("--cate_impute_type", type=str, default="backward", choices=["linear", "backward",
                                                                                     "forward"])

    # note data
    parser.add_argument("--notes_order", type=str, default='Last')
    parser.add_argument("--num_of_notes", type=int, default=20)
    parser.add_argument("--model_name", type=str, default="bioLongformer")
    parser.add_argument("--max_length", type=int, default=512)

    # optim
    parser.add_argument("--bert_opt", type=str, default="AdamW")
    parser.add_argument("--other_opt", type=str, default="AdamW")
    parser.add_argument("--class_opt", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--other_lr", type=float, default=0.0001)
    parser.add_argument("--bert_lr", type=float, default=0.00001)
    parser.add_argument("--class_lr", type=float, default=0.0005)

    # sche
    parser.add_argument("--bert_sche", type=str, default="linear_with_warmup")
    parser.add_argument("--other_sche", type=str, default="None")
    parser.add_argument("--class_sche", type=str, default="None")
    parser.add_argument("--warm_up", type=float, default=0.08)

    # pretrain model save
    parser.add_argument("--freeze", type=str, default="Bert", choices=["Bert+Other", "Bert", "Other"])
    parser.add_argument("--pretrain_model", type=str,
                        default="./save/pretrain/exp_12_new_structure_impute&(mtand&mvand)_0527_seed_42_percent_1.0_epoch_30_batch_3_bioLongformer512_bert_update_2_10_notenum_4_emb_dim_128_bert_lr_2e-05_other_lr_0.0004/18.pth")

    # train
    parser.add_argument("--num_train_epochs", type=int, default=40)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--modeltype", type=str, default='TS_Text')
    parser.add_argument("--before_update_bert_epochs", type=int, default=0)
    parser.add_argument("--num_update_bert_epochs", type=int, default=2)
    parser.add_argument("--bertcount", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    # eval
    parser.add_argument("--eval", type=bool, default=True)  # TODO: False:不包含eval
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)

    # test
    parser.add_argument("--full_model", type=str,
                        default="./save/48ihm/  /9.pth")

    # model
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--ts_dim", type=int, default=128)
    parser.add_argument("--ts_embed_dim", type=int, default=128)
    parser.add_argument("--remove_rep", type=str, default="None", choices=['None', 'time', 'type', 'density', 'invar'])
    parser.add_argument("--mixup_level", type=str, default="batch_seq",
                        choices=['batch', 'batch_seq', 'batch_seq_feature'])
    parser.add_argument("--mtand_mvand_mixup_level", type=str, default="batch_seq",
                        choices=['batch_seq', 'batch_seq_feature'])

    # classifier
    parser.add_argument("--classifier_nhead", type=int, default=4)
    parser.add_argument("--classifier_layers", type=int, default=2)

    # ts model--dual attention
    parser.add_argument("--ts_model_backbone", type=str, default="DualAttention", choices=['DualAttention', 'Ori'])
    parser.add_argument("--ts_model_wo_part", type=str, default="None",
                        choices=["None", "mTAND", "mFAND", "Impute", "feature_gate"])
    parser.add_argument("--ts_dual_attention_layer", type=int, default=2)
    parser.add_argument("--full_attn", action='store_true')
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument('--d_k', type=int, default=8)
    parser.add_argument('--d_v', type=int, default=8)

    # note model--attention
    parser.add_argument("--bert_rep", type=str, default="CLS", choices=["CLS", "Avg-pooling"])
    parser.add_argument("--note_rep", type=str, default="val+time+tau", choices=["val", "val+time", "val+time+tau"])
    parser.add_argument("--note_n_head", type=int, default=8)
    parser.add_argument('--note_d_k', type=int, default=32)
    parser.add_argument('--note_d_v', type=int, default=32)

    # mtand
    parser.add_argument("--ts_mtand_value_transfer", type=bool, default=False)
    parser.add_argument("--note_mtand_value_transfer", type=bool, default=False)

    # model restore
    parser.add_argument("--ts_restore_len", type=int, default=3)  # l

    # model contrastive learning
    parser.add_argument("--contrastive_num", type=int, default=3)  # L_t

    parser.add_argument("--dropout", type=float, default=0.1)

    # Loss = CONT_loss + lambda_1 * ts_restore_mse + lambda_2 * note_restore_cos
    parser.add_argument(
        "--loss",
        default="CONT+RESTORE",
        choices=["CONT", "RESTORE", "CONT+RESTORE"],
    )
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--learnable_temp", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--linear_decompos", type=bool, default=False)
    parser.add_argument("--lambda_1", type=float, default=1.0)
    parser.add_argument("--lambda_2", type=float, default=1.0)

    return parser.parse_args()