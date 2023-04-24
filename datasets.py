# -*- coding: utf-8 -*-
# @Time    : 9/12/2022 3:16 PM
# @Author  : Gang Qu
# @FileName: datasets.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import pandas as pd
import utils as ut
import dgl
from dgl.data import DGLDataset
import time
from tqdm import trange
from dgl.data.utils import save_graphs, load_graphs


class PNCdataset(Dataset):
    def __init__(self, file_name):
        file = np.load(os.path.join('F:\projects\AttentionGCNReorg\data', file_name+'.npz'))
        self.fmri = file['fmri'].astype(np.float32)
        self.age = file['age'].squeeze().astype(int)
        self.gender = file['gender'].squeeze().astype(int)
        self.wrat = file['wrat'].squeeze().astype(np.float32)
        self.pvrt = file['pvrt'].squeeze().astype(np.float32)
        self.pmat = file['pmat'].squeeze().astype(np.float32)

    def __len__(self):
        return self.fmri.shape[0]

    def __getitem__(self, idx):
        fmri = np.corrcoef(self.fmri[idx, :, :])
        age = self.age[idx]
        gender = self.gender[idx]
        wrat = self.wrat[idx]
        pvrt = self.pvrt[idx]
        pmat = self.pmat[idx]
        return fmri, age, gender, wrat, pvrt, pmat


class dglPNCdataset(Dataset):
    def __init__(self, file_name, pos_enc_dim=207, use_cache=True, sparse=20, gl=False, age_thre=16, postfix=None):
        self.file_name = file_name
        self.mni, self.mni_dist, self.roi_index = ut.MNI_space()
        self.mni_dist = torch.tensor(self.mni_dist)
        self.mni = torch.tensor(self.mni)
        if postfix:
            save_name = 'dglPNCDataset' + '_' + file_name + '_' + 'knn' + str(sparse) + '_' + 'gl' + '_' + str(
                gl) + '_'+ postfix+ '.bin'
        else:
            save_name = 'dglPNCDataset' + '_' + file_name + '_' + 'knn' + str(sparse) + '_' + 'gl' + '_' + str(gl) + '.bin'
        graph_file_name = os.path.join('F:\projects\AttentionGCNReorg\data', save_name)
        print('initial dataset')
        if use_cache and os.path.exists(graph_file_name):
            self.graphs, label_dict = load_graphs(graph_file_name)
            self.y_ages = label_dict['age']
            self.y_genders = label_dict['gender']
            self.y_scores = label_dict['score']
        else:
            self.data = PNCdataset(self.file_name)
            self.pos_enc_dim = pos_enc_dim
            self.y_scores = []
            self.y_genders = []
            self.y_ages = []
            self.graphs = []

            for idx in trange(len(self.data)):
                data = self.data[idx]

                fmri, age, gender, wrat, pvrt, pmat = data

                if age > age_thre * 12:
                    fmri = torch.tensor(fmri.astype(np.float32))
                    age = torch.tensor(age)
                    gender = torch.tensor(gender)
                    wrat = torch.tensor(wrat).reshape(-1, 1)
                    pvrt = torch.tensor(pvrt).reshape(-1, 1)
                    pmat = torch.tensor(pmat).reshape(-1, 1)
                    y_score = torch.cat((wrat, pvrt, pmat), dim=0)

                    y_gender = gender
                    y_age = age
                    try:
                        x, edge_list, edge_weight = ut.get_graph_data(fmri, e=self.mni_dist, method='pearson',
                                                                      sparse=sparse)
                    except:
                        continue


                    self.y_scores.append(y_score.reshape(1, -1))
                    self.y_genders.append(y_gender.reshape(1, -1))
                    self.y_ages.append(y_age.reshape(1, -1))
                    g = ut.encode_graph(x, edge_list, edge_weight, pos_enc_dim, self.mni)
                    self.graphs.append(g)
                else:
                    pass
            self.y_ages = torch.cat(self.y_ages, dim=0)
            self.y_genders = torch.cat(self.y_genders, dim=0)
            self.y_scores = torch.cat(self.y_scores, dim=0)
            graph_labels = {'score': self.y_scores, 'gender': self.y_genders, 'age': self.y_ages}
            save_graphs(graph_file_name, self.graphs, graph_labels)
        print('Finish initialization!')

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        y_score = self.y_scores[idx]
        y_age = self.y_ages[idx]
        sample = {'g': g, 'y_score': y_score, 'y_age': y_age, 'mni': self.mni, 'mni_dist': self.mni_dist, 'roi_index': self.roi_index}

        return sample


def dglPNC_collect_func(batch_dic):
    g = []
    y_score = []
    y_age = []
    for i in range(len(batch_dic)):
        batch_i = batch_dic[i]
        g.append(batch_i['g'])
        y_score.append(batch_i['y_score'].squeeze())
        y_age.append(batch_i['y_age'].squeeze())
    res = {}
    res['g'] = dgl.batch(g)
    res['y_score'] = torch.stack(y_score)
    res['y_age'] = torch.stack(y_age)
    res['mni'] = torch.Tensor(batch_dic[0]['mni'])
    res['mni_dist'] = torch.Tensor(batch_dic[0]['mni_dist'])
    res['roi_index'] = torch.Tensor(batch_dic[0]['roi_index'])
    return res


class HCPdataset(Dataset):
    def __init__(self, file_name):
        file = np.load(os.path.join('F:\projects\AttentionGCNReorg\data', file_name + '.npz'), allow_pickle=True)
        self.fmri = file['fmri'].astype(np.float32)
        gender = []
        for i in file['gender'].squeeze():
            if i.item() == 'M':
                gender.append(0)
            else:
                gender.append(1)
        self.pmat = file['pmat'].squeeze().astype(np.float32)
        self.gender = np.array(gender)

    def __len__(self):
        return self.fmri.shape[0]

    def __getitem__(self, idx):
        fmri = np.corrcoef(self.fmri[idx, :, :])
        gender = self.gender[idx]
        pmat = self.pmat[idx]
        return fmri, gender, pmat


class dglHCPdataset(Dataset):
    def __init__(self, file_name, pos_enc_dim=207, use_cache=True, sparse=20, gl=False):
        self.file_name = file_name
        self.mni, self.mni_dist, self.roi_index = ut.MNI_space_HCP()
        self.mni_dist = torch.tensor(self.mni_dist)
        self.mni = torch.tensor(self.mni)
        save_name = 'dglHCPDataset' + '_' + file_name + '_' + 'knn' + str(sparse) + '_' + 'gl' + '_' + str(gl) + '.bin'
        graph_file_name = os.path.join('F:\projects\AttentionGCNReorg\data', save_name)
        print('initial dataset')
        if use_cache and os.path.exists(graph_file_name):
            self.graphs, label_dict = load_graphs(graph_file_name)
            self.y_genders = label_dict['gender']
            self.y_scores = label_dict['score']
        else:
            self.data = HCPdataset(self.file_name)
            self.pos_enc_dim = pos_enc_dim
            self.y_scores = []
            self.y_genders = []
            self.graphs = []

            for idx in trange(len(self.data)):
                data = self.data[idx]
                fmri, gender, pmat = data

                fmri = torch.tensor(fmri.astype(np.float32))
                gender = torch.tensor(gender)
                pmat = torch.tensor(pmat).reshape(-1, 1)
                y_score = pmat

                y_gender = gender

                x, edge_list, edge_weight = ut.get_graph_data(fmri, e=self.mni_dist, method='pearson',
                                                           sparse=sparse)

                self.y_scores.append(y_score.reshape(1, -1))
                self.y_genders.append(y_gender.reshape(1, -1))

                g = ut.encode_graph(x, edge_list, edge_weight, pos_enc_dim, self.mni)
                self.graphs.append(g)

            self.y_genders = torch.cat(self.y_genders, dim=0)
            self.y_scores = torch.cat(self.y_scores, dim=0)
            graph_labels = {'score': self.y_scores, 'gender': self.y_genders}
            save_graphs(graph_file_name, self.graphs, graph_labels)
        print('Finish initialization!')

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        y_score = self.y_scores[idx]

        sample = {'g': g, 'y_score': y_score, 'mni': self.mni, 'mni_dist': self.mni_dist, 'roi_index': self.roi_index}

        return sample

def dglHCP_collect_func(batch_dic):
    g = []
    y_score = []
    for i in range(len(batch_dic)):
        batch_i = batch_dic[i]
        g.append(batch_i['g'])
        y_score.append(batch_i['y_score'].squeeze())
    res = {}
    res['g'] = dgl.batch(g)
    res['y_score'] = torch.stack(y_score)
    res['mni'] = torch.Tensor(batch_dic[0]['mni'])
    res['mni_dist'] = torch.Tensor(batch_dic[0]['mni_dist'])
    res['roi_index'] = torch.Tensor(batch_dic[0]['roi_index'])
    return res

if __name__ == '__main__':
    from models.layers import GatedGCNLayer, GatedGCNLSPELayer
    from models.GGT import GCN, GatedGCNNet
    import json
    import time
    seed = 110
    from utils import seed_it
    seed_it(seed)
    with open('F:\projects\AttentionGCNReorg\configs\GatedAttentionGCN_emoid_multiregression.json') as f:
        config = json.load(f)
    print(config['net_params'])
    mydataset = dglPNCdataset(file_name='rest', pos_enc_dim=config['net_params']['input_dim'] - 3,
                              sparse=30)
    # print(len(mydataset))
    train_size = int(0.7 * len(mydataset))
    val_size = int(0.1 * len(mydataset))
    test_size = len(mydataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(mydataset, [train_size, val_size, test_size])
    # print(len(train_set))
    # print(len(val_set))
    # print(len(test_set))
    t0 = time.time()
    for i in torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=64, collate_fn=dglPNC_collect_func, num_workers=12):
        print(i)
        print(i['g'])
        print('all', time.time() - t0)
        t0 = time.time()
