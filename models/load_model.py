# -*- coding: utf-8 -*-
# @Time    : 10/15/2022 3:23 PM
# @Author  : Gang Qu
# @FileName: load_model.py

from models.GCN import ChebNet
from models.GGT import GGTNet
from models.GGCN import GGCNNet
from models.SAN import SAN
from models.GAT import GAT
from models.GT import GraphTransformerNet
from models.LR import LR
from models.MLP import MLP
def load_model(config):
    if config['model'] == 'GCN':
        model = ChebNet(config['net_params'])
    elif config['model'] == 'GGCN':
        model = GGCNNet(config['net_params'])
    elif config['model'] == 'GGT':
        model = GGTNet(config['net_params'])
    elif config['model'] == 'SAN':
        model = SAN(config['net_params'])
    elif config['model'] == 'GAT':
        model = GAT(config['net_params'])
    elif config['model'] == 'GT':
        model = GraphTransformerNet(config['net_params'])
    elif config['model'] == 'LR':
        model = LR(config['net_params'])
    elif config['model'] == 'MLP':
        model = MLP(config['net_params'])
    return model


if __name__ == '__main__':
    import json
    with open('F:\projects\AttentionGCNReorg\configs\GatedAttentionGCN_nback_multiregression.json') as f:
        config = json.load(f)
    model = load_model(config)
    print(model)