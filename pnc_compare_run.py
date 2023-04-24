# -*- coding: utf-8 -*-
# @Time    : 3/21/2023 2:55 PM
# @Author  : Gang Qu
# @FileName: pnc_compare_run.py
import os
paradigms = ['rest_pnc', 'emoid_pnc', 'nback_pnc']
paradigms = ['rest_pnc']
models = ['LR', 'MLP', 'GCN', 'GGCN', 'GT', 'GAT', 'SAN']
models = ['SAN']
if __name__ == '__main__':
    for paradigm in paradigms:
        for model in models:
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_PNC_multiregression.py --paradigms {p} --max_epochs {e} --cnb_scores all --config {m}_PNCmultiregression'.format(
                p=paradigm, e=int(100), m=model)
            os.system(cmd)
