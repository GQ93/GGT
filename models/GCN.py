# -*- coding: utf-8 -*-
# @Time    : 3/9/2023 12:03 AM
# @Author  : Gang Qu
# @FileName: GCN.py

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers import MLPReadout



class ChebNet(nn.Module):
    def __init__(self, net_params):
        super(ChebNet, self).__init__()
        in_feats = net_params['input_dim']
        hidden_feats = net_params['hidden_dim']
        out_feats = net_params['output_dim']
        num_layers = net_params['L']
        k = net_params['k']
        self.task_loss = net_params['task_loss']
        self.readout = net_params['readout']

        self.k = k

        self.layers = nn.ModuleList()
        self.layers.append(dgl.nn.ChebConv(in_feats, hidden_feats, k))
        for _ in range(num_layers - 2):
            self.layers.append(dgl.nn.ChebConv(hidden_feats, hidden_feats, k))
        self.layers.append(dgl.nn.ChebConv(hidden_feats, out_feats, k))
        self.MLP_layer = MLPReadout(out_feats, net_params['predict_out'])

    def forward(self, h, e=None, p=None, g=None):
        # g: DGLGraph batched graph, h: node features tensor
        with g.local_scope():
            # Set the node features for each graph in the batch
            g.ndata['h'] = h
            # try:
            #     laplacian = torch.tensor(np.array(dgl.laplacian_lambda_max(g)).astype(np.float32)).to(h.device)
            # except:
            #     laplacian = torch.tensor(np.array([2]).astype(np.float32)).to(h.device)


            # Apply Chebyshev polynomial filters
            for i in range(len(self.layers)):
                h = self.layers[i](g, h)

                if i != len(self.layers) - 1:
                    h = F.relu(h)

            # Max pooling over nodes in each graph in the batch
            g.ndata['h'] = h
            if self.readout == "sum":
                hg = dgl.sum_nodes(g, 'h')
            elif self.readout == "max":
                hg = dgl.max_nodes(g, 'h')
            elif self.readout == "mean":
                hg = dgl.mean_nodes(g, 'h')
            else:
                hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            hg = hg.reshape(hg.shape[0], -1)
            return self.MLP_layer(hg), []

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
