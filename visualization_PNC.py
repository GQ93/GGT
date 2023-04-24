# -*- coding: utf-8 -*-
# @Time    : 11/9/2022 9:05 PM
# @Author  : Gang Qu
# @FileName: visualization_PNC.py
import argparse
import utils
import os
from os.path import join as pj
from os.path import abspath
import json
import datasets
import torch
from models.load_model import load_model
from train import PNC_multi_regression
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
        'Cyan': (0, 'cyan', 'Sensory/somatomotor'),
        'Orange': (0, 'cyan', 'Sensory/somatomotor'),
        'Purple': (1, 'purple', 'Cingulo-opercular Task Control'),
        'Pink': (2, 'pink', 'Auditory'),
        'Red': (3, 'red', 'Default mode'),
        'Gray': (4, 'grey', 'Memory retrieval'),
        'Blue': (5, 'blue', 'Visual'),
        'Yellow': (6, 'yellow', 'Fronto-parietal Task Control'),
        'Black': (7, 'black', 'Salience'),
        'Brown': (8, 'brown', 'Subcortical'),
        'Teal': (9, 'green', 'Attention'),
        'Green': (9, 'green', 'Attention'),
        'Pale blue': (10, 'steelblue', 'Cerebellar')
    }
    labels = ['Sensory/somatomotor', 'Cingulo-opercular Task Control', 'Auditory', 'Default Mode', 'Memory Retrieval',
               'Visual', 'Fronto-parietal Task Control', 'Salience', 'Subcortical', 'Ventral Attention', 'Dorsal Attention', 'Cerebellar']

    network_index = [(1, 35), (36, 49), (50, 62), (63, 120), (121, 125), (126, 156), (157, 181), (182, 199), (200, 212),
                     (213, 221), (222, 232), (233, 236)]

    color_names = ['cyan', 'purple', 'olive', 'red', 'grey', 'blue', 'yellow', 'deepskyblue', 'lime', 'teal', 'green', 'steelblue']

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
    if args.dataset == 'dglPNC':
        dataset = datasets.dglPNCdataset(file_name=args.paradigms, pos_enc_dim=config['net_params']['pos_enc_dim'] - 3,
                                         sparse=args.sparse)
    print(len(dataset))
    # split the dataset and define the dataloader
    if args.dataset == 'dglPNC':
        data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size,
                                                   collate_fn=datasets.dglPNC_collect_func)

    # define the model
    model = load_model(config)
    model.to(device)

    if config['pretrain']:
        checkpoint = torch.load(abspath(pj('results', config['pretrain_model_name'])))
        model.load_state_dict(checkpoint['model'])

    _, _, roi_info = utils.MNI_space(dist=False, networks=True)
    # eval
    _, attention = PNC_multi_regression.evaluate_network(model, device, data_loader, epoch=0, weight_score=config['net_params']["weight_score"])
    heat_map = 0
    vec = 0
    for att in attention:
        att_value = att[0].cpu().numpy().squeeze()

        att_row, att_col = att[2]
        att_row = att_row.cpu().numpy()
        att_col = att_col.cpu().numpy()
        heat_map_i = ss.csr_matrix((att_value, (att_row, att_col)),
                          shape = (264, 264))
        heat_map += heat_map_i.todense()

        vec += att[1].cpu().numpy()
        # heat_map += att[0].cpu().numpy()
        # vec += att[1].cpu().numpy()

    heat_map /= len(attention)
    vec /= len(attention)

    is_symmetric = np.array_equal(heat_map, heat_map.T)
    if not is_symmetric:
        heat_map = (heat_map + heat_map.T)/2
    print(np.array_equal(heat_map, heat_map.T))
    roi_reorder = [int(x-1) for x in roi_info[0]]

    heat_map = heat_map.reshape(264, 264)
    # remove self-loop
    mask = np.ones(264) - np.identity(264)

    heat_map = heat_map[roi_reorder, :]
    heat_map = heat_map[:, roi_reorder]
    heat_map = (heat_map - np.min(heat_map)) / (np.max(heat_map) - np.min(heat_map))

    # row_sums = heat_map.sum(axis=1)
    # heat_map = heat_map * row_sums[:, np.newaxis]
    # col_sums = heat_map.sum(axis=0)
    # heat_map = heat_map * col_sums[np.newaxis, :]
    heat_map = heat_map[:236, :236]
    # heat_map = (heat_map - np.mean(heat_map, axis=1, keepdims=True))/np.std(heat_map, axis=1, keepdims=True)
    is_symmetric = np.array_equal(heat_map, heat_map.T)
    print("Is tensor_sym symmetric? ", is_symmetric)
    with open(r'F:\projects\AttentionGCNReorg\results\figures\heatmap_pnc.npy', 'wb') as f:
        np.save(f, heat_map)
    fig, ax = plt.subplots()
    im = ax.imshow(heat_map, cmap='hot', interpolation='nearest')
    network_name_pool = ['Sensory/somatomotor', 'Auditory', 'Default mode', 'Cerebellar', 'Visual']
    for network_name in network_patches:
        x, y = network_patches[network_name]
        color = patche_colors[network_name]
        xy_min = x - 1
        width = y - x
        print(xy_min)
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
    plt.savefig(os.path.join(save_pth, 'all_' + args.paradigms + '_KNN' + str(args.sparse) + '.png'),
                 bbox_inches='tight', pad_inches=0.05)
    plt.show()

    # vec = np.asarray(vec, dtype='float64')
    #
    #
    #
    #
    # roi_idx = []
    # c = []
    #
    #
    # for i, info in enumerate(roi_info[1]):
    #     if info in c_dict:
    #         roi_idx.append(i)
    #         c.append(c_dict[info][0])
    #
    # pos_embedded = TSNE(n_components=2, init='pca', perplexity=10).fit_transform(vec[roi_idx])
    # fig, ax = plt.subplots()
    # # scatter = plt.scatter(x, y, c=values, cmap=colors)
    # scatter = plt.scatter(pos_embedded[roi_idx, 0], pos_embedded[roi_idx, 1], c=c, cmap=colors)
    # plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    # plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PNC multi-regression')
    parser.add_argument('--batch_size', default=4, help='batch size')
    parser.add_argument('--dropout', default=None, help='dropout rate')
    parser.add_argument('--config', default='GGT_PNCmultiregression_visualization', help='config file name')
    parser.add_argument('--dataset', default='dglPNC', help='dataset name')
    parser.add_argument('--sparse', default=30, help='sparsity for knn graph')
    parser.add_argument('--gl', default=False, help='graph learning beta')
    parser.add_argument('--paradigms', default='rest_pnc',
                        choices=[
                            'emoid_pnc', 'nback_pnc', 'rest_pnc',
                        ],
                        help='fmri paradigms')
    args = parser.parse_args()

    main(args)