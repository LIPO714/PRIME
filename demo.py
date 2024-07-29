
weight_decay = 0.01

optimizer_weight_dc = []
optimizer_no_weight_dc = []
optimizer_new_weight_dc = []
optimizer_new_no_weight_dc = []

no_decay = ["bias", "LayerNorm.weight"]  # 不加decay的部分
skip = {}
if hasattr(model, "no_weight_decay"):
    skip = model.no_weight_decay()  # 模型可能不需要decay的部分

for n, p in model.named_parameters():  # name param
    if 'bert' in n:  # 如果是bert的
        if any(nd in n for nd in no_decay) or len(p.shape) == 1 or n.endswith(".bias") or n in skip:
            optimizer_no_weight_dc.append(p)
        else:
            optimizer_weight_dc.append(p)
    else:
        if any(nd in n for nd in no_decay) or len(p.shape) == 1 or n.endswith(".bias") or n in skip:
            optimizer_new_no_weight_dc.append(p)
        else:
            optimizer_new_weight_dc.append(p)


optimizer_grouped_parameters = [
    {
        "params": optimizer_weight_dc,
        "weight_decay": weight_decay,
    },
    {
        "params": optimizer_no_weight_dc,
        "weight_decay": 0.0,
    },
]

optimizer_new_parameters = [
    {
        "params": optimizer_new_weight_dc,
        "weight_decay": weight_decay,
    },
    {
        "params": optimizer_new_no_weight_dc,
        "weight_decay": 0.0,
    },
]

optimizer= torch.optim.Adam([
                {'params': optimizer_new_parameters},
                {'params': optimizer_grouped_parameters, 'lr': args.txt_learning_rate}
            ], lr=args.ts_learning_rate)

optimizer= torch.optim.Adam([
                {'params': optimizer_new_parameters, 'lr': args.ts_learning_rate},
                {'params': optimizer_grouped_parameters, 'lr': args.txt_learning_rate}
])