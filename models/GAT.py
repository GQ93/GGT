# -*- coding: utf-8 -*-
# @Time    : 3/9/2023 12:03 AM
# @Author  : Gang Qu
# @FileName: GAT.py
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import MLPReadout

class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_fc = nn.Linear(2 * out_feats, 1, bias=False)

    def edge_attention(self, edges):
        z = torch.cat([edges.src['z'], edges.dst['z']], dim=-1)
        a = self.attn_fc(z)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, h):
        z = self.fc(h).view(-1, self.num_heads, self.out_feats)
        g.ndata['z'] = z

        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)

        return g.ndata.pop('h')


class GAT(nn.Module):
    def __init__(self, net_params):
        super(GAT, self).__init__()
        in_feats = net_params['input_dim']
        hidden_feats = net_params['hidden_dim']
        out_feats = net_params['output_dim']
        num_heads = net_params['num_heads']
        num_layers = net_params['L']
        self.task_loss = net_params['task_loss']
        self.readout = net_params['readout']
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(in_feats, hidden_feats, num_heads))
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(num_heads * hidden_feats, hidden_feats, num_heads))
        self.layers.append(GATLayer(num_heads * hidden_feats, out_feats, 1))
        self.MLP_layer = MLPReadout(out_feats * num_heads, net_params['predict_out'])

    def forward(self, h, e=None, p=None, g=None):
        if not g:
            g = dgl.rand_graph(int(h.shape[0]), int(e.shape[0]), device=h.device)
        for i, layer in enumerate(self.layers):

            h = layer(g, h)

            if i != len(self.layers) - 1:
                h = F.elu(h)

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
