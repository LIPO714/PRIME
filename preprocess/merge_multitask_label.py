'''
Merge downstream task datasets.
Merge the label of "ihm" into "phe"
'''
import pickle

import numpy as np
import pandas as pd


def read_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def diff_float(time1, time2):
    # compute time2-time1
    # return differences in hours but as float
    h = (time2-time1).astype('timedelta64[m]').astype(int)
    return h/60.0


def generate_48ihm_label(ihm_data):
    label = {}
    for i in range(len(ihm_data)):
        now_name = ihm_data[i]['name']
        label[now_name] = ihm_data[i]['label']
    return label


def generate_los_decom_label(now_name, root_dir):
    patient_id = now_name.split('_')[0]
    episode_id = int(now_name.split('_')[1][7:]) - 1
    stays_file = root_dir + patient_id + '/stays.csv'
    df = pd.read_csv(stays_file)
    in_time = df['INTIME'].tolist()
    in_time = in_time[episode_id]
    death_time = df['DEATHTIME'].tolist()
    death_time = death_time[episode_id]
    los = df['LOS'].tolist()
    los = float(los[episode_id]) * 24

    # print("name:", now_name)
    # print("intime:", in_time)
    # print("deathtime:", death_time)
    # print(type(death_time))
    # print("los:", los)

    if isinstance(death_time, float):  # death_time == nan
        decom = -1
    else:
        in_time = np.datetime64(in_time)
        death_time = np.datetime64(death_time)
        decom = diff_float(in_time, death_time)
        if los > decom:
            los = decom

    # print("decom:", decom)
    # print("los:", los)

    age = df['AGE'].tolist()
    age = float(age[episode_id])
    gender_str = df['GENDER'].tolist()
    if gender_str[episode_id] == "M":  # male
        gender = 1
    else:  # female
        gender = 0
    demographics = [age, gender]

    return los, decom, demographics


def merge_multitask_label_to_pheno(phe_data, ihm_label, root_dir):
    for i in range(len(phe_data)):
        now_name = phe_data[i]["name"]
        # ihm
        if now_name in ihm_label.keys():
            phe_data[i]['ihm_label'] = ihm_label[now_name]  # 是否死亡
            phe_data[i]['ihm_task'] = 1
        else:
            phe_data[i]['ihm_task'] = 0

        # los and decompensation; demographics(基本信息：age, gender)
        los, decom_time, demographics = generate_los_decom_label(now_name, root_dir)  

        phe_data[i]['los'] = los
        phe_data[i]['decom_time'] = decom_time
        phe_data[i]['demographics'] = demographics

        pheno_label = phe_data[i].pop('label')
        phe_data[i]['pheno_label'] = pheno_label  # 表型分类

    return phe_data


if __name__ == '__main__':
    dataset = "test"  # "test"  "train"
    ihm_dir = "./data/ihm/"
    phe_dir = "./data/pheno/"
    root_dir = "./data/root/"
    out_dir = "./data/representing_learning/"

    ihm_data = read_pkl(ihm_dir + dataset + "ts_note_data.pkl")
    phe_data = read_pkl(phe_dir + dataset + "ts_note_data.pkl")

    ihm_label = generate_48ihm_label(ihm_data)

    phe_data = merge_multitask_label_to_pheno(phe_data, ihm_label, root_dir + dataset + '/')

    with open(out_dir + dataset + "_ts_note_data.pkl", 'wb') as f:
        pickle.dump(phe_data, f)
