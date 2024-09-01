# -*- coding: utf-8 -*-
# @Time    : 8/10/2023 7:23 PM
# @Author  : Gang Qu
# @FileName: compare_k_run.py
import os
k = [10, 20, 30, 40, 50, 60, 70]
k = [ 40]
cnb_scores = ['all']
paradigms_hcp = ['rest_hcp']
paradigms_pnc = ['rest_pnc']
for paradigm in paradigms_hcp:
    for k_i in k:
        cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_HCP_regression.py --paradigms {p} --max_epochs {e} --sparse {s}'.format(
            p=paradigm, e=int(5), s=k_i)
        os.system(cmd)

for paradigm in paradigms_pnc:
    for cnb_score in cnb_scores:
        for k_i in k:
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_PNC_multiregression.py --paradigms {p} --max_epochs {e} --cnb_scores {c} --sparse {s}'.format(
                p=paradigm, e=int(5 ), c=cnb_score, s=k_i)
            os.system(cmd)
