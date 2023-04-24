# -*- coding: utf-8 -*-
# @Time    : 3/24/2023 10:12 PM
# @Author  : Gang Qu
# @FileName: MLP.py



import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers import MLPReadout



class MLP(nn.Module):
    def __init__(self, net_params):
        super(MLP, self).__init__()
        in_feats = net_params['input_dim'] * (net_params['input_dim'] + 1) // 2
        hidden_feats = net_params['hidden_dim']
        out_feats = net_params['output_dim']
        num_layers = net_params['L']
        self.task_loss = net_params['task_loss']
        self.readout = net_params['readout']


        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, hidden_feats))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_feats, hidden_feats))
        self.layers.append(nn.Linear(hidden_feats, out_feats))
        self.MLP_layer = MLPReadout(out_feats, net_params['predict_out'])

    def forward(self, h, e=None, p=None, g=None):
        # g: DGLGraph batched graph, h: node features tensor
        batch_size = g.batch_size
        h = h.view(batch_size, h.shape[-1], h.shape[-1])
        indices = torch.triu_indices(row=h.shape[1], col=h.shape[2])
        h = h[:, indices[0], indices[1]]
        for i in range(len(self.layers)):
            h = self.layers[i](h)

            if i != len(self.layers) - 1:
                h = F.relu(h)

        return self.MLP_layer(h), []

    def loss(self, scores, targets, weights=[0.1, 0.55, 0.35], experiment='PNC'):
        if experiment == 'PNC':
            if self.task_loss == 'L1':
                # weights = [0.1, 0.55, 0.35]
                loss = (weights[0] * nn.L1Loss()(scores[:, 0], targets[:, 0])
                        + weights[1] * nn.L1Loss()(scores[:, 1], targets[:, 1])
                        + weights[2] * nn.L1Loss()(scores[:, 2], targets[:, 2]))
            elif self.task_loss == 'L2':
                # weights = [0.1, 0.5, 0.4]
                loss = (weights[0] * nn.MSELoss()(scores[:, 0], targets[:, 0])
                        + weights[1] * nn.MSELoss()(scores[:, 1], targets[:, 1])
                        + weights[2] * nn.MSELoss()(scores[:, 2], targets[:, 2]))
        elif experiment == 'HCP':
            if self.task_loss == 'L1':
                loss = nn.L1Loss()(scores.squeeze(), targets.squeeze())

            elif self.task_loss == 'L2':
                loss = nn.MSELoss()(scores.squeeze(), targets.squeeze())

        return loss