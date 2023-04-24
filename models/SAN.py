# -*- coding: utf-8 -*-
# @Time    : 3/8/2023 11:56 PM
# @Author  : Gang Qu
# @FileName: SAN.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np
from models.layers import MLPReadout
"""
    Graph Transformer Layer

"""

"""
    Util functions
"""

def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}

    return func


# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """

    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

    return func


def exp_real(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5)) / (L + 1)}

    return func


def exp_fake(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': L * torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5)) / (L + 1)}

    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, gamma, in_dim, out_dim, num_heads, dropout, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = gamma
        self.dropout = dropout

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.E = nn.Linear(in_dim, out_dim * num_heads, bias=True)

            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)

        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.E = nn.Linear(in_dim, out_dim * num_heads, bias=False)


            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):


        real_ids = g.edges(form='eid')

        try:
            del g.edata['score']
        except:
            pass

        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'), edges=real_ids)


        # scale scores by sqrt(d)
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        # Use available edge features to modify the scores for edges
        g.apply_edges(imp_exp_attn('score', 'E'), edges=real_ids)

        g.apply_edges(exp('score'), edges=real_ids)

        # Send weighted values to target nodes
        eids = g.edges()
        try:
            del g.ndata['wV']
        except:
            pass
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_soft', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))

    def forward(self, g, h, e):

        Q_h = self.Q(h)
        K_h = self.K(h)
        E = self.E(e)


        V_h = self.V(h)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.edata['E'] = E.view(-1, self.num_heads, self.out_dim)

        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        h_out = F.dropout(h_out, self.dropout, training=self.training)

        return h_out


class GraphTransformerLayer(nn.Module):
    """
        Param:
    """

    def __init__(self, gamma, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True,
                 residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(gamma, in_dim, out_dim // num_heads, num_heads, dropout, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, e):
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h_attn_out = self.attention(g, h, e)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)
class SAN(nn.Module):
    def __init__(self, net_params):
        super().__init__()

        num_atom_type = net_params['input_dim']
        num_bond_type = net_params['num_edge_feat']


        gamma = net_params['gamma']

        GT_layers = net_params['L']
        GT_hidden_dim = net_params['hidden_dim']
        GT_out_dim = net_params['output_dim']
        GT_n_heads = net_params['num_heads']
        predict_out = net_params['predict_out']

        self.residual = net_params['residual']
        self.readout = net_params['readout']
        self.task_loss = net_params['task_loss']

        dropout = net_params['dropout']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']




        self.embedding_h = nn.Linear(num_atom_type, GT_hidden_dim)
        self.embedding_e = nn.Linear(num_bond_type, GT_hidden_dim)

        self.layers = nn.ModuleList([GraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, dropout,
                                                           self.layer_norm, self.batch_norm, self.residual) for
                                     _ in range(GT_layers - 1)])

        self.layers.append(
            GraphTransformerLayer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, dropout,
                                  self.layer_norm, self.batch_norm, False))
        self.MLP_layer = MLPReadout(GT_out_dim, predict_out)  # 1 out dim since regression problem

    def forward(self, h, e, p=None, g=None):
        if not g:
            g = dgl.rand_graph(int(h.shape[0]), int(e.shape[0]), device=h.device)
        # input embedding
        h = self.embedding_h(h)
        e = self.embedding_e(e)

        # GNN
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return self.MLP_layer(hg), []


    def loss(self, scores, targets, weights=[0.1, 0.55, 0.35], experiment='PNC'):
        # Loss A: Task loss -------------------------------------------------------------
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
                # weights = [0.1, 0.55, 0.35]
                loss = nn.L1Loss()(scores.squeeze(), targets.squeeze())

            elif self.task_loss == 'L2':
                # weights = [0.1, 0.5, 0.4]
                loss = nn.MSELoss()(scores.squeeze(), targets.squeeze())

        return loss