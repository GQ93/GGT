# -*- coding: utf-8 -*-
# @Time    : 8/16/2022 8:50 PM
# @Author  : Gang Qu
# @FileName: layers.py

import numpy as np
import utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl

class GGCNLayer(nn.Module):
    """
        Param: []
    """

    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True) # edge feature
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def forward(self, h, e, p=None, g=None):
        if not g:
            g = dgl.rand_graph(int(h.shape[0]), int(e.shape[0]), device=h.device)
        h_in = h  # for residual connection
        e_in = e  # for residual connection

        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['e'] = e
        g.edata['Ce'] = self.C(e)

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)

        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization
            e = self.bn_node_e(e)  # batch normalization

        h = F.relu(h)  # non-linear activation
        e = F.relu(e)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h, e, g

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                                            self.in_channels,
                                                            self.out_channels)

class GGCNLSPELayer(nn.Module):
    """
        Param: []
    """

    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A1 = nn.Linear(input_dim * 2, output_dim, bias=True)
        self.A2 = nn.Linear(input_dim * 2, output_dim, bias=True)
        self.B1 = nn.Linear(input_dim, output_dim, bias=True)
        self.B2 = nn.Linear(input_dim, output_dim, bias=True)
        # self.B3 = nn.Linear(input_dim, output_dim, bias=True)
        self.B3 = nn.Linear(input_dim, output_dim, bias=True)
        self.C1 = nn.Linear(input_dim, output_dim, bias=True)
        self.C2 = nn.Linear(input_dim, output_dim, bias=True)

        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
        # self.bn_node_p = nn.BatchNorm1d(output_dim)


    def message_func_for_vij(self, edges):
        hj = edges.src['h']  # h_j
        pj = edges.src['p']  # p_j
        vij = self.A2(torch.cat((hj, pj), -1))
        return {'v_ij': vij}

    def message_func_for_pj(self, edges):
        pj = edges.src['p']  # p_j
        return {'C2_pj': self.C2(pj)}

    def compute_normalized_eta(self, edges):
        return {'eta_ij': edges.data['sigma_hat_eta'] / (
                    edges.dst['sum_sigma_hat_eta'] + 1e-6)}  # sigma_hat_eta_ij/ sum_j' sigma_hat_eta_ij'

    def forward(self, h, e, p, g=None):

        if not g:
            g = dgl.rand_graph(int(h.shape[0]), int(e.shape[0]), device=h.device)

        with g.local_scope():

            # for residual connection
            h_in = h
            p_in = p
            e_in = e

            # For the h's
            g.ndata['h'] = h
            g.ndata['A1_h'] = self.A1(torch.cat((h, p), -1))
            # self.A2 being used in message_func_for_vij() function
            g.ndata['B1_h'] = self.B1(h)
            g.ndata['B2_h'] = self.B2(h)

            # For the p's
            g.ndata['p'] = p
            g.ndata['C1_p'] = self.C1(p)
            # self.C2 being used in message_func_for_pj() function

            # For the e's
            g.edata['e'] = e
            g.edata['B3_e'] = self.B3(e)

            # --------------------------------------------------------------------------------------#
            # Calculation of h
            g.apply_edges(fn.u_add_v('B1_h', 'B2_h', 'B1_B2_h'))
            g.edata['hat_eta'] = g.edata['B1_B2_h'] + g.edata['B3_e']
            g.edata['sigma_hat_eta'] = torch.sigmoid(g.edata['hat_eta'])
            g.update_all(fn.copy_e('sigma_hat_eta', 'm'), fn.sum('m', 'sum_sigma_hat_eta'))  # sum_j' sigma_hat_eta_ij'
            g.apply_edges(self.compute_normalized_eta)  # sigma_hat_eta_ij/ sum_j' sigma_hat_eta_ij'
            g.apply_edges(self.message_func_for_vij)  # v_ij
            g.edata['eta_mul_v'] = g.edata['eta_ij'] * g.edata['v_ij']  # eta_ij * v_ij
            g.update_all(fn.copy_e('eta_mul_v', 'm'), fn.sum('m', 'sum_eta_v'))  # sum_j eta_ij * v_ij
            g.ndata['h'] = g.ndata['A1_h'] + g.ndata['sum_eta_v']

            # Calculation of p
            g.apply_edges(self.message_func_for_pj)  # p_j
            g.edata['eta_mul_p'] = g.edata['eta_ij'] * g.edata['C2_pj']  # eta_ij * C2_pj
            g.update_all(fn.copy_e('eta_mul_p', 'm'), fn.sum('m', 'sum_eta_p'))  # sum_j eta_ij * C2_pj
            g.ndata['p'] = g.ndata['C1_p'] + g.ndata['sum_eta_p']

            # --------------------------------------------------------------------------------------#

            # passing towards output
            h = g.ndata['h']
            p = g.ndata['p']
            e = g.edata['hat_eta']

            # batch normalization
            if self.batch_norm:
                h = self.bn_node_h(h)
                e = self.bn_node_e(e)
                # No BN for p

            # non-linear activation
            h = F.relu(h)
            e = F.relu(e)
            p = torch.tanh(p)

            # residual connection
            if self.residual:
                h = h_in + h
                p = p_in + p
                e = e_in + e

                # dropout
            h = F.dropout(h, self.dropout, training=self.training)
            p = F.dropout(p, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)

            return h, e, p, g

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                                            self.in_channels,
                                                            self.out_channels)

class GGTLayer(nn.Module):
    """
        Param: []
    """

    def __init__(self, input_dim, output_dim, dropout, batch_norm, num_heads=1, merge='cat', residual=False,
                 att_act1='tanh', att_act2='sigmoid', eps=1e-12):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.num_heads = num_heads
        self.merge = merge
        self.eps = eps

        att_act = {
            'leaky_relu': torch.nn.functional.leaky_relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'relu': torch.relu
        }
        self.att_act1 = att_act[att_act1] if att_act1 in att_act else torch.relu
        self.att_act2 = att_act[att_act2] if att_act2 in att_act else torch.relu

        if input_dim != output_dim:
            self.residual = False

        # self.Q = nn.ModuleDict()
        self.K = nn.ModuleDict()
        self.V = nn.ModuleDict()
        self.layer_norm = nn.LayerNorm(output_dim)

        for i in range(num_heads):
            # self.Q['Q' + str(i)] = nn.Linear(input_dim * 2, output_dim, bias=True)
            self.K['K' + str(i)] = nn.Linear(input_dim * 2, output_dim, bias=True)
            self.V['V' + str(i)] = nn.Linear(input_dim * 2, output_dim, bias=True)

        self.A = nn.Linear(input_dim * 2, output_dim, bias=True)
        self.B1 = nn.Linear(input_dim, output_dim, bias=True)
        self.B2 = nn.Linear(input_dim, output_dim, bias=True)
        # self.B3 = nn.Linear(input_dim, output_dim, bias=True)
        self.B3 = nn.Linear(input_dim, output_dim, bias=True)
        self.C1 = nn.Linear(input_dim, output_dim, bias=True)
        self.C2 = nn.Linear(input_dim, output_dim, bias=True)

        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
        # self.bn_node_p = nn.BatchNorm1d(output_dim)

    def message_func_for_vij(self, edges):
        return {'v_ij': edges.src['V_h']}

    def message_func_for_pj(self, edges):
        pj = edges.src['p']  # p_j
        return {'C2_pj': self.C2(pj)}

    def compute_normalized_eta(self, edges):
        return {'eta_ij': edges.data['sigma_hat_eta'] / (
                    edges.dst['sum_sigma_hat_eta'] + self.eps)}  # sigma_hat_eta_ij/ sum_j' sigma_hat_eta_ij'

    def compute_attention_scores(self, edges):
        # return {'alpha_ij': edges.data['simQK'] / (edges.dst['sum_simQK'])}  # simQK_ij / sum_j' simQK_ij'
        return {'alpha_ij': edges.data['simQK']}  # simQK_ij

    def forward(self, h, e, p, g=None):
        if not g:
            g = dgl.rand_graph(int(h.shape[0]), int(e.shape[0]), device=h.device)
        with g.local_scope():

            # for residual connection
            h_in = h
            p_in = p
            e_in = e

            # # For the h's

            v = []
            q = []
            k = []
            for head in range(self.num_heads):
                v.append(self.V['V' + str(head)](torch.cat((h, p), -1)))
                # q.append(self.Q['Q' + str(head)](torch.cat((h, p), -1)))
                q.append(self.K['K' + str(head)](torch.cat((h, p), -1)))
                k.append(self.K['K' + str(head)](torch.cat((h, p), -1)))
            if self.merge == 'cat':
                # concat on the output feature dimension (dim=1)
                vh = torch.cat(v, dim=1)
                qh = torch.cat(q, dim=1)
                kh = torch.cat(k, dim=1)
            else:
                # merge using average
                vh = torch.mean(torch.stack(v))
                qh = torch.mean(torch.stack(q))
                kh = torch.mean(torch.stack(k))
            g.ndata['Q_h'] = qh
            g.ndata['V_h'] = vh
            g.ndata['K_h'] = kh


            # g.ndata['h'] = h
            # g.ndata['A_h'] = self.A(torch.cat((h, p), -1))
            g.ndata['A_h'] = vh
            del v, q, k, qh, vh, kh
            # self.V being used in message_func_for_vij() function
            g.ndata['B1_h'] = self.B1(h)
            g.ndata['B2_h'] = self.B2(h)

            # For the p's
            g.ndata['p'] = p
            g.ndata['C1_p'] = self.C1(p)
            # self.C2 being used in message_func_for_pj() function

            # For the e's
            g.edata['e'] = e
            g.edata['B3_e'] = self.B3(e)

            # --------------------------------------------------------------------------------------#
            # Calculation of h
            g.apply_edges(fn.u_add_v('B1_h', 'B2_h', 'B1_B2_h'))
            g.edata['hat_eta'] = g.edata['B1_B2_h'] + g.edata['B3_e']
            g.edata['sigma_hat_eta'] = torch.sigmoid(g.edata['hat_eta'])
            g.update_all(fn.copy_e('sigma_hat_eta', 'm'), fn.sum('m', 'sum_sigma_hat_eta'))  # sum_j' sigma_hat_eta_ij'

            g.apply_edges(self.compute_normalized_eta)  # sigma_hat_eta_ij/ sum_j' sigma_hat_eta_ij'
            g.apply_edges(self.message_func_for_vij)  # v_ij

            # g.ndata['sigmaQ'] = torch.exp(torch.tanh(g.ndata['Q_h']))
            # g.ndata['sigmaK'] = torch.exp(torch.sigmoid(g.ndata['K_h']))
            g.ndata['sigmaQ'] = torch.exp(self.att_act1(g.ndata['Q_h']))
            g.ndata['sigmaK'] = torch.exp(self.att_act2(g.ndata['K_h']))

            g.apply_edges(fn.u_dot_v('sigmaQ', 'sigmaK', 'simQK'))

            # g.update_all(fn.copy_e('simQK', 'm'), fn.sum('m', 'sum_simQK'))
            g.apply_edges(self.compute_attention_scores)
            g.edata['eta_mul_v'] = g.edata['eta_ij'] * g.edata['alpha_ij'] * g.edata['v_ij']  # eta_ij * v_ij
            g.update_all(fn.copy_e('eta_mul_v', 'm'), fn.sum('m', 'sum_eta_v'))  # sum_j eta_ij * v_ij
            g.ndata['h'] = g.ndata['A_h'] + g.ndata['sum_eta_v']



            # Calculation of p
            g.apply_edges(self.message_func_for_pj)  # p_j
            g.edata['eta_mul_p'] = g.edata['eta_ij'] * g.edata['C2_pj']  # eta_ij * C2_pj
            g.update_all(fn.copy_e('eta_mul_p', 'm'), fn.sum('m', 'sum_eta_p'))  # sum_j eta_ij * C2_pj
            g.ndata['p'] = g.ndata['C1_p'] + g.ndata['sum_eta_p']

            # --------------------------------------------------------------------------------------#

            # passing towards output
            h = g.ndata['h']
            p = g.ndata['p']
            e = g.edata['hat_eta']



            # batch normalization
            if self.batch_norm:
                h = self.bn_node_h(h)
                e = self.bn_node_e(e)
                # No BN for p

            # non-linear activation
            h = F.relu(h)
            e = F.relu(e)
            p = torch.tanh(p)

            # residual connection
            if self.residual:
                h = h_in + h
                p = p_in + p
                e = e_in + e

            h = self.layer_norm(h)
            # dropout
            h = F.dropout(h, self.dropout, training=self.training)
            p = F.dropout(p, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)

            g_new = g.cpu()
            self.attention = g.edata['alpha_ij']
            return h, e, p, g_new

    # def __repr__(self):
    #     return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
    #                                                         self.in_channels,
    #                                                         self.out_channels)

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout=0.0):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        h = self.dropout(h)
        return {'h': h}



class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, dropout=0.0, L=1):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.dropout = dropout

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = F.dropout(y, self.dropout, training=self.training)
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y