# -*- coding: utf-8 -*-
# @Time    : 10/11/2022 8:25 PM
# @Author  : Gang Qu
# @FileName: GGT.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from scipy import sparse as sp
from scipy.sparse.linalg import norm

"""
    GatedGCN and GatedGCN-LSPE

"""
from models.layers import GGCNLayer, GGCNLSPELayer,  GGTLayer, MLPReadout





class GGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_edge_feat = net_params['num_edge_feat']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['output_dim']
        dropout = net_params['dropout']
        self.task_loss = net_params['task_loss']
        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.pe_init = net_params['pe_init']

        self.use_lapeig_loss = net_params['use_lapeig_loss']
        self.lambda_loss = net_params['lambda_loss']
        self.alpha_loss = net_params['alpha_loss']
        self.pos_enc_dim = net_params['pos_enc_dim']
        if 'MNI' in net_params:
            self.MNI = net_params['MNI']
        else:
            self.MNI = True
        if 'PE' in net_params:
            self.MNI = net_params['PE']
        else:
            self.MNI = True
        self.embedding_h = nn.Linear(net_params['input_dim'], hidden_dim)

        if not 'MNI':
            self.pos_enc_dim -= 3
        if not self.MNI and self.PE:
            self.pos_enc_dim -= 3
        elif not self.PE and self.MNI:
            self.pos_enc_dim = 3

        if self.pe_init in ['rand_walk', 'lap_pe']:
            self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)

        self.embedding_e = nn.Linear(num_edge_feat, hidden_dim)

        if self.pe_init == 'rand_walk':
            # LSPE
            self.layers = nn.ModuleList([GGCNLSPELayer(hidden_dim, hidden_dim, dropout,
                                                           self.batch_norm, residual=self.residual) for _ in
                                         range(self.n_layers - 1)])
        else:
            # NoPE or LapPE
            self.layers = nn.ModuleList([GGCNLayer(hidden_dim, hidden_dim, dropout,
                                                       self.batch_norm, residual=self.residual) for _
                                         in range(self.n_layers - 1)])

        self.MLPReadout(out_dim, net_params['predict_out'])

        if self.pe_init == 'rand_walk':
            self.p_out = nn.Linear(hidden_dim, out_dim)
            self.Whp = nn.Linear(out_dim + hidden_dim, out_dim)

        self.g = None

    def forward(self, h, e, p, g=None):
        """

        :param h: node features
        :type h: tensor
        :param e: edge features
        :type e: tensor
        :param p: position vectors
        :type p: tensor
        :param g: dgl graph
        :type g: dgl.graph.dgl.batchedgraph
        :return:
        :rtype:
        """
        if not self.MNI and self.PE:
            p = p[:, :-3]
        elif not self.PE and self.MNI:
            p = p[:, -3:]
        elif not self.MNI and not self.PE:
            p = torch.ones_like(p)
        if not g:
            g = dgl.rand_graph(int(h.shape[0]), int(e.shape[0]), device=h.device)
        h = self.embedding_h(h)     # input_dim -> hidden_dim
        if self.pe_init in ['rand_walk', 'lap_pe']:
            p = self.embedding_p(p)     # pos_enc_dim -> hidden_dim
        if self.pe_init == 'lap_pe':
            h = h + p
            p = None


        if not self.edge_feat:
            e = torch.ones(e.size(0), 2).to(h.device)
        e = self.embedding_e(e)     # num_edge_features -> hidden_dim

        # convnets
        gs = []
        for conv in self.layers:
            h, e, p, g_rev = conv(h=h, e=e, p=p, g=g)  # hidden_dim -> hidden_dim
            gs.extend(dgl.unbatch(g_rev))

        del g_rev
        g.ndata['h'] = h

        if self.pe_init == 'rand_walk':
            # Implementing p_g = p_g - torch.mean(p_g, dim=0)
            p = self.p_out(p)   # hidden_dim -> output_dim
            g.ndata['p'] = p
            means = dgl.mean_nodes(g, 'p')
            batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
            p = p - batch_wise_p_means

            # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            g.ndata['p'] = p
            g.ndata['p2'] = g.ndata['p'] ** 2
            norms = dgl.sum_nodes(g, 'p2')
            norms = torch.sqrt(norms)
            batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)
            p = p / batch_wise_p_l2_norms
            g.ndata['p'] = p

            # Concat h and p
            hp = self.Whp(torch.cat((g.ndata['h'], g.ndata['p']), dim=-1))  # hidden_dim + pos_enc_dim
            g.ndata['h'] = hp

        # readout
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        self.g = g  # For util; To be accessed in loss() function

        return self.MLP_layer(hg), gs

    def loss(self, scores, targets, weights=[0.1, 0.55, 0.35]):
        # Loss A: Task loss -------------------------------------------------------------
        if self.task_loss == 'L1':
            # weights = [0.1, 0.55, 0.35]
            loss_a = (weights[0] * nn.L1Loss()(scores[:, 0], targets[:, 0])
                      + weights[1] * nn.L1Loss()(scores[:, 1], targets[:, 1])
                      + weights[2] * nn.L1Loss()(scores[:, 2], targets[:, 2]))
        elif self.task_loss == 'L2':
            # weights = [0.1, 0.5, 0.4]
            loss_a = (weights[0] * nn.MSELoss()(scores[:, 0], targets[:, 0])
                      + weights[1] * nn.MSELoss()(scores[:, 1], targets[:, 1])
                      + weights[2] * nn.MSELoss()(scores[:, 2], targets[:, 2]))

        if self.use_lapeig_loss:
            # Loss B: Laplacian Eigenvector Loss --------------------------------------------
            g = self.g
            n = g.number_of_nodes()

            # Laplacian
            A = g.adjacency_matrix(scipy_fmt="csr")
            N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
            L = sp.eye(n) - N * A * N

            p = g.ndata['p']
            pT = torch.transpose(p, 1, 0)
            loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(scores.device)), p))

            # Correct batch-graph wise loss_b_2 implementation; using a block diagonal matrix
            bg = dgl.unbatch(g)
            batch_size = len(bg)
            P = sp.block_diag([bg[i].ndata['p'].detach().cpu() for i in range(batch_size)])
            PTP_In = P.T * P - sp.eye(P.shape[1])
            loss_b_2 = torch.tensor(norm(PTP_In, 'fro') ** 2).float().to(scores.device)

            loss_b = (loss_b_1 + self.lambda_loss * loss_b_2) / (self.pos_enc_dim * batch_size * n)

            del bg, P, PTP_In, loss_b_1, loss_b_2

            loss = loss_a + self.alpha_loss * loss_b

        else:
            loss = loss_a

        return loss


class GGTNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_edge_feat = net_params['num_edge_feat']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['output_dim']
        dropout = net_params['dropout']
        self.task_loss = net_params['task_loss']
        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.pe_init = net_params['pe_init']

        self.use_lapeig_loss = net_params['use_lapeig_loss']
        self.lambda_loss = net_params['lambda_loss']
        self.alpha_loss = net_params['alpha_loss']
        self.pos_enc_dim = net_params['pos_enc_dim']
        if 'att_act1' in net_params:
            self.att_act1 = net_params['att_act1']
        else:
            self.att_act1 = 'tanh'

        if 'att_act2' in net_params:
            self.att_act2 = net_params['att_act2']
        else:
            self.att_act2 = 'sigmoid'

        self.embedding_h = nn.Linear(net_params['input_dim'], hidden_dim)



        if self.pe_init in ['rand_walk', 'lap_pe']:
            self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)
        self.embedding_e = nn.Linear(num_edge_feat, hidden_dim)

        if self.pe_init == 'rand_walk':
            # LSPE
            self.layers = nn.ModuleList([GGTLayer(hidden_dim, hidden_dim, dropout,
                                                           self.batch_norm, residual=self.residual,
                                                                    att_act1=self.att_act1,
                                                                    att_act2=self.att_act2) for _ in
                                         range(self.n_layers - 1)])
        else:
            # NoPE or LapPE
            self.layers = nn.ModuleList([GGCNLayer(hidden_dim, hidden_dim, dropout,
                                                       self.batch_norm, residual=self.residual) for _
                                         in range(self.n_layers - 1)])

        self.MLP_layer = MLPReadout(out_dim, net_params['predict_out'], dropout)

        if self.pe_init == 'rand_walk':
            self.p_out = nn.Linear(hidden_dim, out_dim)
            self.Whp = nn.Linear(out_dim + hidden_dim, out_dim)

        self.g = None

    def forward(self, h, e, p, g=None):
        """
        :param g: dgl graph
        :type g: dgl.graph.dgl.batchedgraph
        :param h: node features
        :type h: tensor
        :param p: position vectors
        :type p: tensor
        :param e: edge features
        :type e: tensor
        :return:
        :rtype:
        """
        if not g:
            g = dgl.rand_graph(int(h.shape[0]), int(e.shape[0]), device=h.device)
        h = self.embedding_h(h)     # input_dim -> hidden_dim
        if self.pe_init in ['rand_walk', 'lap_pe']:
            p = self.embedding_p(p)     # pos_enc_dim -> hidden_dim

        if self.pe_init == 'lap_pe':
            h = h + p
            p = None

        if not self.edge_feat:  # edge feature set to 1
            e = torch.ones(e.size(0), 2).to(h.device)
        e = self.embedding_e(e)     # num_edge_features -> hidden_dim

        # convnets
        gs = []
        for lth, conv in enumerate(self.layers):
            h, e, p, g_rev = conv(h=h, e=e, p=p, g=g)  # hidden_dim -> hidden_dim
            if lth == 0:
                gs.extend(dgl.unbatch(g_rev))
        del g_rev

        g.ndata['h'] = h

        if self.pe_init == 'rand_walk':
            # Implementing p_g = p_g - torch.mean(p_g, dim=0)
            p = self.p_out(p)   # hidden_dim -> output_dim
            g.ndata['p'] = p
            means = dgl.mean_nodes(g, 'p')
            batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
            p = p - batch_wise_p_means

            # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            g.ndata['p'] = p
            g.ndata['p2'] = g.ndata['p'] ** 2
            norms = dgl.sum_nodes(g, 'p2')
            norms = torch.sqrt(norms)
            batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)
            p = p / batch_wise_p_l2_norms
            g.ndata['p'] = p

            # Concat h and p
            hp = self.Whp(torch.cat((g.ndata['h'], g.ndata['p']), dim=-1))  # hidden_dim + pos_enc_dim
            g.ndata['h'] = hp

        # readout
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        self.g = g  # For util; To be accessed in loss() function

        return self.MLP_layer(hg), gs

    def loss(self, scores, targets, weights=[0.1, 0.55, 0.35], experiment='PNC'):
        # Loss A: Task loss -------------------------------------------------------------
        if experiment == 'PNC':
            if self.task_loss == 'L1':
                # weights = [0.1, 0.55, 0.35]
                loss_a = (weights[0] * nn.L1Loss()(scores[:, 0], targets[:, 0])
                          + weights[1] * nn.L1Loss()(scores[:, 1], targets[:, 1])
                          + weights[2] * nn.L1Loss()(scores[:, 2], targets[:, 2]))
            elif self.task_loss == 'L2':
                # weights = [0.1, 0.5, 0.4]
                loss_a = (weights[0] * nn.MSELoss()(scores[:, 0], targets[:, 0])
                          + weights[1] * nn.MSELoss()(scores[:, 1], targets[:, 1])
                          + weights[2] * nn.MSELoss()(scores[:, 2], targets[:, 2]))
        elif experiment == 'HCP':
            if self.task_loss == 'L1':
                # weights = [0.1, 0.55, 0.35]
                loss_a = nn.L1Loss()(scores.squeeze(), targets.squeeze())

            elif self.task_loss == 'L2':
                # weights = [0.1, 0.5, 0.4]
                loss_a = nn.MSELoss()(scores.squeeze(), targets.squeeze())



        if self.use_lapeig_loss:
            # Loss B: Laplacian Eigenvector Loss --------------------------------------------
            g = self.g
            n = g.number_of_nodes()

            # Laplacian
            A = g.adjacency_matrix(scipy_fmt="csr")
            N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
            L = sp.eye(n) - N * A * N

            p = g.ndata['p']
            pT = torch.transpose(p, 1, 0)
            loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(scores.device)), p))

            # Correct batch-graph wise loss_b_2 implementation; using a block diagonal matrix
            bg = dgl.unbatch(g)
            batch_size = len(bg)
            P = sp.block_diag([bg[i].ndata['p'].detach().cpu() for i in range(batch_size)])
            PTP_In = P.T * P - sp.eye(P.shape[1])
            loss_b_2 = torch.tensor(norm(PTP_In, 'fro') ** 2).float().to(scores.device)

            loss_b = (loss_b_1 + self.lambda_loss * loss_b_2) / (self.pos_enc_dim * batch_size * n)
            # print('b12', loss_b_1, loss_b_2)
            del bg, P, PTP_In, loss_b_1, loss_b_2

            loss = loss_a + self.alpha_loss * loss_b
            # print('a,b', loss_a, loss_b)
        else:
            loss = loss_a

        return loss