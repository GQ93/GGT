# -*- coding: utf-8 -*-
# @Time    : 3/21/2023 6:33 PM
# @Author  : Gang Qu
# @FileName: hcp_compare_run.py
import os
# paradigms = ['language_hcp', 'gambling_hcp', 'wm_hcp',
#              'emoid_hcp']
paradigms = ['social_hcp', 'relational_hcp', 'moto_hcp', 'language_hcp', 'gambling_hcp', 'wm_hcp',
                            'emoid_hcp', 'rest_hcp']
paradigms = ['wm_hcp']
models = ['LR', 'GCN', 'GGCN', 'GT', 'GAT', 'SAN', 'GGT']
models = [ 'GGCN']
if __name__ == '__main__':
    for paradigm in paradigms:
        for model in models:
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_HCP_regression.py --paradigms {p} --max_epochs {e} --config {m}_HCPregression_PMAT'.format(
                p=paradigm, e=int(100), m=model)
            os.system(cmd)
