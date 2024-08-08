# PRIME: Pretraining Model for Patient Representation Learning Using Irregular Multimodal Electronic Health Records

This repository contains the PyTorch implementation for the paper ”PRIME: Pretraining Model for Patient Representation Learning Using Irregular Multimodal Electronic Health Records“.

### Environment

Run the following commands to create a conda environment:

```
conda create -n PRIME python=3.7.16
source activate PRIME
pip install -r requirements.txt
```

### Data

Our multimodal dataset is derived from MIMIC-III, a publicly available real-world electronic health record (EHR) dataset that encompasses both monitoring data and clinical notes. We processed the data following:

1. **Download data from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/).**

   This dataset is a restricted-access resource. To access the files, you must be a credentialed user and sign the data use agreement (DUA) for the project. Because of the DUA, we cannot provide the data directly.

2. **Generate multivariate irregular time series (MITS) data following [MIMIC-III Benchmarks](https://github.com/YerevaNN/mimic3-benchmarks).**

   There are five downstream tasks: in-hospital-mortality, decompensation, length-of-stay, phenotyping and multitask. We conduct experiments on in-hospital-mortality and phenotyping.

3. **Generate irregular clinical notes (INS) data following [ClinicalNotesICU](https://github.com/kaggarwal/ClinicalNotesICU).**

4. **Merge two modality data.**

   We first following [MultimodalMIMIC](https://github.com/XZhang97666/MultimodalMIMIC?tab=readme-ov-file) to merge the two modalities of data. However, in contrast to the end-to-end task in  [MultimodalMIMIC](https://github.com/XZhang97666/MultimodalMIMIC?tab=readme-ov-file), our pretraining task retain a larger volume of data. Consequently, we updated the `preprocessing.py` and the `read_all_text_append_json` function in the `text_util.py`, with the revised code located in `./preprocess/update_code_in_MultimodalMIMIC`. 

   Subsequently, we merged the data from the two task datasets into a comprehensive dataset by `python ./preprocess/merge_multitask_label.py`. This process resulted in the creation of `train_ts_note_data.pkl` and `test_ts_note_data.pkl`.

5. **Generate datasets for `pretrain`, `48IHM`, and `24PHENO` tasks.**

   Move `train_ts_note_data.pkl` and `test_ts_note_data.pkl` to `./data/`.

   Pretrain:

   ```python
   python ./preprocess/preprocess_pretrain_data.py
   ```

   48IHM:

   ```python
   python ./preprocess/preprocess_48ihm.py
   ```

   24PHENO:

   ```python
   python ./preprocess/preprocess_24pheno.py
   ```

​	Choose the parameter `percent` in `preprocess_48ihm.py` and `preprocess_24pheno.py` from `[1.0, 0.1, 0.01]` to generate datasets with labeled data of different proportions.



Finally, the file tree in `./data/` should be:

```
 The file tree should be:
.
├── 24pheno
│   ├── test_24pheno_0.01.pkl
│   ├── test_24pheno_0.1.pkl
│   ├── test_24pheno_1.0.pkl
│   ├── train_24pheno_0.01.pkl
│   ├── train_24pheno_0.1.pkl
│   ├── train_24pheno_1.0.pkl
│   ├── val_24pheno_0.01.pkl
│   ├── val_24pheno_0.1.pkl
│   └── val_24pheno_1.0.pkl
├── 48ihm
│   ├── test_48ihm_0.01.pkl
│   ├── test_48ihm_0.1.pkl
│   ├── test_48ihm_1.0.pkl
│   ├── train_48ihm_0.01.pkl
│   ├── train_48ihm_0.1.pkl
│   ├── train_48ihm_1.0.pkl
│   ├── val_48ihm_0.01.pkl
│   ├── val_48ihm_0.1.pkl
│   └── val_48ihm_1.0.pkl
├── channel_std.py
├── metadata.json
├── pretrain.pkl
├── test_ts_note_data.pkl
└── train_ts_note_data.pkl
```

### Pretraining

Our pretraining task was conducted using two NVIDIA GeForce RTX 3090 GPUs, running two scripts sequentially to start two processes.

```shell
./pretrain_1.sh
./pretrain_2.sh
```

If executed on a single GPU, the following command can be used. However, this approach reduces the number of negative samples, which may affect performance.

```python
python main.py --init_method tcp://localhost:26699 -g 0 --rank 0 --world_size 1
```

### **Fine-tuning**

Run the following commands to fine-tune:

```
python downstream_task.py --init_method tcp://localhost:26699 -g 0 --rank 0 --world_size 1
```

Change different configuration files in `downstream.py` to change subtasks:

```python
# TODO: choose downstream tasks: 24PHENO or 48IHM
# from all_exps.downstream_24PHENO_args_config import parse_args  # 24PHENO
from all_exps.downstream_48IHM_args_config import parse_args  # 48IHM
```

### Document Description

`Clinical-Longformer` stores Bert models.
`all_exps` contains configuration files.
