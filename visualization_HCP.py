# -*- coding: utf-8 -*-
# @Time    : 2/16/2023 7:00 PM
# @Author  : Gang Qu
# @FileName: visualization_HCP.py
import argparse
import utils
import os
from os.path import join as pj
from os.path import abspath
import json
import datasets
import torch
from models.load_model import load_model
from train import HCP_regression
import time
import torch.optim.lr_scheduler as lr_scheduler
from dgl import RemoveSelfLoop
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import utils
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import scipy.sparse as ss


def main(args):
    c_dict = {
        'Cyan': (0, 'cyan', 'Medial Frontal'),
        'Purple': (1, 'purple', 'Frontoparietal'),
        'Pink': (2, 'olive', 'Default Mode'),
        'Red': (3, 'red', 'Subcortical-cerebellum'),
        'Gray': (4, 'grey', 'Motor'),
        'Blue': (5, 'blue', 'Visual I'),
        'Yellow': (6, 'yellow', 'Visual II'),
        'Black': (7, 'deepskyblue', 'Visual Association'),
    }
    labels = ['Medial Frontal', 'Frontoparietal', 'Default Mode', 'Subcortical-cerebellum',
              'Motor', 'Visual I', 'Visual II', 'Visual Association']

    network_index = [(1, 29), (30, 63), (64, 83), (84, 173), (174, 223), (224, 241), (242, 250), (251, 268)]

    color_names = ['cyan', 'purple', 'olive', 'red', 'grey', 'blue', 'yellow', 'deepskyblue']

    colors = ListedColormap(color_names)

    network_patches = dict(zip(labels, network_index))
    patche_colors = dict(zip(labels, color_names))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # import config files and update args
    config_file = pj(abspath('configs'), args.config + '.json')
    with open(config_file) as f:
        config = json.load(f)
    config['dataset'] = args.dataset
    if args.dropout:
        config['net_params']['dropout'] = args.dropout

    # define dataset
    if args.dataset == 'dglHCP':
        dataset = datasets.dglHCPdataset(file_name=args.paradigms, pos_enc_dim=config['net_params']['pos_enc_dim'] - 3,
                                         sparse=args.sparse)
    print(len(dataset))

    # split the dataset and define the dataloader
    if args.dataset == 'dglHCP':
        data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size,
                                                  collate_fn=datasets.dglHCP_collect_func)

    # define the model
    model = load_model(config)
    model.to(device)

    if config['pretrain']:
        checkpoint = torch.load(abspath(pj('results', config['pretrain_model_name'])))
        model.load_state_dict(checkpoint['model'])

    _, _, roi_info = utils.MNI_space_HCP(dist=False, networks=True)
    print(roi_info)
    # eval
    _, attention = HCP_regression.evaluate_network(model, device, data_loader, epoch=0,
                                                   weight_score=config['net_params']["weight_score"])
    heat_map = 0
    vec = 0
    for att in attention:
        att_value = att[0].cpu().numpy().squeeze()

        att_row, att_col = att[2]
        att_row = att_row.cpu().numpy()
        att_col = att_col.cpu().numpy()

        heat_map_i = ss.csr_matrix((att_value, (att_row, att_col)),
                                   shape=(268, 268))


        heat_map += heat_map_i.todense()
        print(np.array_equal(heat_map, heat_map.T))
        # print(heat_map)

        vec += att[1].cpu().numpy()
        # heat_map += att[0].cpu().numpy()
        # vec += att[1].cpu().numpy()

    heat_map /= len(attention)
    vec /= len(attention)
    # fig, ax = plt.subplots()
    roi_reorder = [int(x - 1) for x in roi_info[0]]

    heat_map = heat_map.reshape(268, 268)

    # remove self-loop
    mask = np.ones((268, 268)) - np.identity(268)
    heat_map = heat_map[roi_reorder, :]
    heat_map = heat_map[:, roi_reorder]
    # heat_map = np.multiply(heat_map,  mask)
    heat_map = (heat_map - np.min(heat_map))/(np.max(heat_map)-np.min(heat_map))
    is_symmetric = np.array_equal(heat_map, heat_map.T)

    # with open(r'F:\projects\AttentionGCNReorg\results\figures\heatmap_hcp.npy', 'wb') as f:
    #     np.save(f, heat_map)

    # row_sums = heat_map.sum(axis=1)
    # heat_map = heat_map * row_sums[:, np.newaxis]
    # col_sums = heat_map.sum(axis=0)
    # heat_map = heat_map * col_sums[np.newaxis, :]
    # heat_map = heat_map[:236, :236]
    # heat_map = (heat_map - np.mean(heat_map, axis=1, keepdims=True))/np.std(heat_map, axis=1, keepdims=True)
    fig, ax = plt.subplots()
    # im = ax.imshow(heat_map * 10, cmap='hot', interpolation='none')
    # heat_map = np.log(heat_map+0.3) - np.log(np.min(heat_map)+0.3)
    im = ax.imshow(heat_map, cmap='hot', interpolation='nearest')
    network_name_pool = ['Sensory/somatomotor', 'Auditory', 'Default mode', 'Cerebellar', 'Visual']
    for network_name in network_patches:
        x, y = network_patches[network_name]
        color = patche_colors[network_name]
        xy_min = x - 1
        width = y - x
        # if network_name in network_name_pool:
        ax.add_patch(
            patches.Rectangle(
                xy=(xy_min, xy_min),
                width=width,
                height=width,
                linewidth=1.5,
                color=color,
                fill=False,
                label=network_name
            )
        )
    plt.legend(prop={'size': 6})
    save_pth = os.path.abspath(r'F:\projects\AttentionGCNReorg\results\figures')
    plt.colorbar(im)
    plt.savefig(os.path.join(save_pth, 'PMAT_' + args.paradigms + '_KNN' + str(args.sparse) + '.png'),
                 bbox_inches='tight', pad_inches=0.05)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HCP multi-regression')
    parser.add_argument('--batch_size', default=4, help='batch size')
    parser.add_argument('--dropout', default=None, help='dropout rate')
    parser.add_argument('--config', default='GGT_HCPregression_visualization', help='config file name')
    parser.add_argument('--dataset', default='dglHCP', help='dataset name')
    parser.add_argument('--sparse', default=30, help='sparsity for knn graph')
    parser.add_argument('--gl', default=False, help='graph learning beta')
    parser.add_argument('--cnb_scores', default='wrat',
                        choices=[
                            'wrat', 'pvrt', 'pmat', 'all'
                        ],
                        help='type of cnb scores')
    parser.add_argument('--paradigms', default='emoid_hcp',
                        choices=[
                            'social_hcp', 'relational_hcp', 'moto_hcp', 'language_hcp', 'gambling_hcp', 'wm_hcp',
                            'emoid_hcp', 'rest_hcp'
                        ],
                        help='fmri paradigms')
    args = parser.parse_args()

    main(args)
