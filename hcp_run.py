# -*- coding: utf-8 -*-
# @Time    : 2/23/2023 11:47 AM
# @Author  : Gang Qu
# @FileName: hcp_run.py
import os
paradigms = ['rest_hcp', 'social_hcp', 'relational_hcp', 'moto_hcp', 'language_hcp', 'gambling_hcp', 'wm_hcp',
             'emoid_hcp']

if __name__ == '__main__':
    for paradigm in paradigms:
        cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_HCP_regression.py --paradigms {p} --max_epochs {e}'.format(
            p=paradigm, e=int(100))
        os.system(cmd)
