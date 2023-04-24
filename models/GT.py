# -*- coding: utf-8 -*-
# @Time    : 3/8/2023 11:56 PM
# @Author  : Gang Qu
# @FileName: GT.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np
from models.layers import MLPReadout
"""
    Graph Transformer Layer with edge features

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


# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}

    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)

        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))

        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))

        # softmax
        g.apply_edges(exp('score'))

        # Send weighted values to target nodes
        eids = g.edges()
        try:
            del g.ndata['wV']
        except:
            pass
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))

        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h, e):

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))  # adding eps to all values here
        e_out = g.edata['e_out']

        return h_out, e_out


class GraphTransformerLayer(nn.Module):
    """
        Param:
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

    def forward(self, h, e, p=None, g=None):
        if not g:
            g = dgl.rand_graph(int(h.shape[0]), int(e.shape[0]), device=h.device)
        h_in1 = h  # for first residual connection
        e_in1 = e  # for first residual connection

        # multi-head attention out

        h_attn_out, e_attn_out = self.attention(g, h, e)


        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)


        h = self.O_h(h)
        e = self.O_e(e)


        if self.residual:
            h = h_in1 + h  # residual connection
            e = e_in1 + e  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h  # for second residual connection
        e_in2 = e  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h  # residual connection
            e = e_in2 + e  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
            super().__init__()
            num_atom_type = net_params['input_dim']
            self.num_bond_type = net_params['num_edge_feat']
            hidden_dim = net_params['hidden_dim']
            num_heads = net_params['num_heads']
            out_dim = net_params['output_dim']
            dropout = net_params['dropout']
            n_layers = net_params['L']
            self.readout = net_params['readout']
            self.layer_norm = net_params['layer_norm']
            self.batch_norm = net_params['batch_norm']
            self.residual = net_params['residual']
            self.edge_feat = net_params['edge_feat']
            self.lap_pos_enc = net_params['lap_pos_enc']
            self.task_loss = net_params['task_loss']
            # max_wl_role_index = 37  # this is maximum graph size in the dataset

            if self.lap_pos_enc:
                pos_enc_dim = net_params['pos_enc_dim']
                self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

            self.embedding_h = nn.Linear(num_atom_type, hidden_dim)

            if self.edge_feat:
                self.embedding_e = nn.Linear(self.num_bond_type, hidden_dim)

            # self.in_feat_dropout = nn.Dropout(in_feat_dropout)

            self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                               self.layer_norm, self.batch_norm, self.residual) for _ in
                                         range(n_layers - 1)])
            self.layers.append(
                GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm,
                                      False))
            self.MLP_layer = MLPReadout(out_dim, net_params['predict_out'])

    def forward(self, h, e, p, g=None):

            # input embedding
            h = self.embedding_h(h)
            if self.lap_pos_enc:
                h_lap_pos_enc = self.embedding_lap_pos_enc(p.float())
                h = h + h_lap_pos_enc
            # if self.wl_pos_enc:
            #     h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            #     h = h + h_wl_pos_enc
            if not self.edge_feat:  # edge feature set to 1
                e = torch.ones(e.size(0), self.num_bond_type).to(h.device)
            e = self.embedding_e(e)

            # convnets
            for conv in self.layers:
                h, e = conv(h, e, p, g)
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