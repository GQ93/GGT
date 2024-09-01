# -*- coding: utf-8 -*-
# @Time    : 8/16/2022 10:42 AM
# @Author  : Gang Qu
# @FileName: utils.py

# utils functions
import os
import torch
import numpy as np
import copy
import logging.config
import random
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy import sparse as sp
import dgl
import torch.nn.functional as F

LOGGING_DIC = {
    'version': 1.0,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format':
                '%(asctime)s %(threadName)s:%(thread)d [%(name)s] %(levelname)s [%(pathname)s:%(lineno)d] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'simple': {
            'format': '%(asctime)s [%(name)s] %(levelname)s %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'test': {
            'format': '%(asctime)s %(message)s',
        },
    },
    'filters': {},
    'handlers': {
        'console_debug_handler': {
            'level': 'DEBUG',  # 日志处理的级别限制
            'class': 'logging.StreamHandler',  # 输出到终端
            'formatter': 'simple'  # 日志格式
        },
        'file_info_handler': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',  # 保存到文件,日志轮转
            'filename': 'user.log',
            'maxBytes': 1024 * 1024 * 10,  # 日志大小 10M
            'backupCount': 10,  # 日志文件保存数量限制
            'encoding': 'utf-8',
            'formatter': 'standard',
        },
        'file_debug_handler': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',  # 保存到文件
            'filename': 'test.log',  # 日志存放的路径
            'encoding': 'utf-8',  # 日志文件的编码
            'formatter': 'test',
        },
    },
    'loggers': {
        'logger1': {  # 导入时logging.getLogger时使用的app_name
            'handlers': ['console_debug_handler'],  # 日志分配到哪个handlers中
            'level': 'DEBUG',
            'propagate': False,
        },
        'logger2': {
            'handlers': ['console_debug_handler', 'file_debug_handler'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}


def adj_to_index(adj, binary=True):
    """
    convert adjacency matrix to edge index
    :param adj: adjacency matrix
    :type adj: array/ tensor
    :param binary: if true return edge weights, if False, return edge index and edge weight
    :type binary: bool
    :return: edge index (edge weight)
    :rtype:
    """
    # adj[adj == 0] += 1e-12
    if isinstance(adj, np.ndarray):
        non_zero_idx = adj.nonzero()
        edge_index = np.ascontiguousarray(np.array(non_zero_idx))
        non_zero_idx = adj.nonzero()
        edge_weight = adj[non_zero_idx]
        if binary:
            return edge_index, np.ones_like(edge_weight)
        else:
            # edge_weight[edge_weight <= 1e-12] = 0
            return edge_index, edge_weight
    elif isinstance(adj, torch.Tensor):
        edge_index = adj.nonzero().t().contiguous()
        non_zero_idx = adj.nonzero(as_tuple=True)
        edge_weight = adj[non_zero_idx]
        if binary:
            return edge_index, torch.ones_like(edge_weight)
        else:
            edge_weight[edge_weight <= 1e-12] = 0
            return edge_index, edge_weight


def get_logger(name, path='results/loggers'):
    """
    get the logger
    :param name: name of logger file
    :type name:
    :return:
    :rtype:
    """
    log_config = copy.deepcopy(LOGGING_DIC)
    if not os.path.exists(path):
        os.makedirs(path)
    log_config['handlers']['file_debug_handler']['filename'] = os.path.join(path, name)
    logging.config.dictConfig(log_config)
    logger = logging.getLogger('logger2')
    return logger


def seed_it(seed):
    """
    set random seed for reproducibility
    :param seed: seed number
    :type seed: int
    :return:
    :rtype:
    """
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def fc_pearson(fmri):
    """
    get fc from fmri
    :param fmri: N_roi * T_time
    :type fmri: tensor
    :return: N * N_roi * N_roi
    :rtype: tensor
    """

    if len(fmri.shape) == 2:
        cor_coef = torch.corrcoef(fmri)
    return cor_coef


def knn_adj(adj, k, e=None):
    """
    keep only k largest entries for each row
    :param adj: N_batch * N_roi * N_roi adjacency matrix
    :type adj: tensor
    :param k: number of entries to be kept
    :type k: int
    :param e: additional edge matrix
    :type e: tensor
    :return:
    :rtype:
    """
    adj_sparse = torch.zeros_like(adj)
    topk, indices = torch.topk(adj, k=k, dim=0)
    adj_sparse[:, :] = adj_sparse[:, :].scatter(0, indices, topk)
    adj_mask = torch.ones_like(adj)
    adj_mask[adj_sparse == 0] = 0
    adj_mask = adj_mask + adj_mask.t()
    adj_mask[adj_mask != 0] = 1
    adj_sparse = adj * adj_mask
    if e is None:
        return adj_sparse
    else:
        e[e == 0] += 1e-12
        e_sparse = torch.ones_like(e)

        # tope = e.gather(0, indices)
        # e_sparse[:, :] = e_sparse[:, :].scatter(0, indices, tope)
        # e_sparse[e_sparse == 1e-12] = 0
        e_sparse *= adj_mask
        return adj_sparse, e_sparse



def get_graph_data(fmri, e=None, method='pearson', sparse=None):
    """
    get and preprocess data
    :param fmri:
    :type fmri:
    :param age:
    :type age:
    :param gender:
    :type gender:
    :param wrat:
    :type wrat:
    :param pvrt:
    :type pvrt:
    :param pmat:
    :type pmat:
    :return:
    :rtype:
    """
    x = fmri
    if method.lower() == 'pearson':
        cor_coef = np.abs(fc_pearson(fmri))
    if e is None:
        if sparse:
            adj_sparse = knn_adj(cor_coef, sparse)
        else:
            adj_sparse = cor_coef
        edge_list, edge_weight = adj_to_index(adj_sparse, binary=False)
    else:
        if sparse:

            adj_sparse, e_sparse = knn_adj(cor_coef, sparse, e)
        else:
            adj_sparse, e_sparse = knn_adj(cor_coef, k=cor_coef.shape[0], e=e)
        edge_list, edge_weight = adj_to_index(adj_sparse, binary=False)
        _, edge_weight_ = adj_to_index(e_sparse, binary=False)
        edge_weight = edge_weight.reshape(-1, 1)
        # edge_weight_[edge_weight_<= 1e-12] = 0
        edge_weight_ = edge_weight_.reshape(-1, 1)
        # print(edge_weight.shape, edge_weight_.shape)
        edge_weight = torch.cat((edge_weight, edge_weight_), dim=1)

    # return x, edge_list, edge_weight, y_score, y_gender, y_age
    return x, edge_list, edge_weight

def encode_graph(x, edge_list, edge_weight, pos_enc_dim, mni):
    """
    :param x:
    :type x:
    :param edge_list:
    :type edge_list:
    :param edge_weight:
    :type edge_weight:
    :return:
    :rtype:
    """
    g = dgl.graph((edge_list[0], edge_list[1]))
    g.ndata['x'] = x
    edge_weight = edge_weight.reshape(g.num_edges(), -1)
    g.edata['x'] = edge_weight

    # add position encoding and MNI coordinates
    g = randwalk_positional_encoding(g, pos_enc_dim)
    g.ndata['p'] = torch.cat((g.ndata['p'], mni), dim=1)
    return g


def MNI_space(template_dir=r'F:\projects\AttentionGCNReorg\data\PP264_template.xls', dist=True, scale=100, networks=False):
    xls = pd.read_excel(template_dir)
    roi_index = xls.iloc[1:, 1].to_numpy(dtype=np.float32)
    x = xls.iloc[1:, 2].to_numpy(dtype=np.float32).reshape(-1, 1)
    y = xls.iloc[1:, 3].to_numpy(dtype=np.float32).reshape(-1, 1)
    z = xls.iloc[1:, 4].to_numpy(dtype=np.float32).reshape(-1, 1)
    mni = np.concatenate((x, y, z), axis=1) / scale
    if networks:
        colors = xls.iloc[1:, 13].values
        networks_names = xls.iloc[1:, 14].values
        roi_index = (roi_index, colors, networks_names)

    if dist:
        distance = euclidean_distances(mni, mni)
        distance += np.identity(distance.shape[0]) * 1e-12
        return mni, distance/np.max(distance), roi_index
    else:
        return mni, None, roi_index


def MNI_space_HCP(template_dir=r'F:\projects\AttentionGCNReorg\data\shen_268_parcellation_networklabels_MNI.csv',
                  dist=True, scale=100, networks=False):
    xls = pd.read_csv(template_dir)
    roi_index = xls.iloc[:, 0].to_numpy(dtype=np.float32)
    x = xls.iloc[:, 4].to_numpy(dtype=np.float32).reshape(-1, 1)
    y = xls.iloc[:, 5].to_numpy(dtype=np.float32).reshape(-1, 1)
    z = xls.iloc[:, 6].to_numpy(dtype=np.float32).reshape(-1, 1)
    mni = np.concatenate((x, y, z), axis=1) / scale
    if networks:
        colors = 0
        networks_names = xls.iloc[:, 8].values
        roi_index = (roi_index, colors, networks_names)

    if dist:
        distance = euclidean_distances(mni, mni)
        distance += np.identity(distance.shape[0]) * 1e-12
        return mni, distance/np.max(distance), roi_index
    else:
        return mni, None, roi_index

def add_eig_vec(g, pos_enc_dim):
    """
     Graph positional encoding v/ Laplacian eigenvectors
     This func is for eigvec visualization, same code as positional_encoding() func,
     but stores value in a diff key 'eigvec'
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['eigvec'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    # zero padding to the end if n < pos_enc_dim
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['eigvec'] = F.pad(g.ndata['eigvec'], (0, pos_enc_dim - n + 1), value=float('0'))

    return g


def lap_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    # g.ndata['eigvec'] = g.ndata['pos_enc']

    return g


def randwalk_positional_encoding(g, pos_enc_dim):
    """
        Initializing positional encoding with RWPE
    """
    # Geometric diffusion features with Random Walk
    A = g.adjacency_matrix(scipy_fmt="csr")
    Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)  # D^-1
    RW = A * Dinv
    M = RW

    # Iterate
    nb_pos_enc = pos_enc_dim
    PE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    for _ in range(nb_pos_enc - 1):
        M_power = M_power * M
        PE.append(torch.from_numpy(M_power.diagonal()).float())
    PE = torch.stack(PE, dim=-1)
    g.ndata['p'] = PE

    return g


def evaluate_mat(predicted, target, method):
    res = predicted - target
    if method == 'RMSE':
        res = res**2
        if isinstance(res, np.ndarray):
            res = np.sqrt(np.sum(res, axis=0)/res.shape[0])
        elif isinstance(res, torch.Tensor):
            res = torch.sqrt(torch.sum(res, dim=0)/res.shape[0]).numpy()
    elif method == 'MAE':
        if isinstance(res, np.ndarray):
            res = np.sum(np.abs(res), axis=0)/res.shape[0]
        elif isinstance(res, torch.Tensor):
            res = (torch.sum(torch.abs(res), dim=0)/res.shape[0]).numpy()

    return res


class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            self.min_delta = validation_loss - train_loss
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0
#
# if __name__ == '__main__':
#     mni, distance = MNI_space()
#     print(type(distance))