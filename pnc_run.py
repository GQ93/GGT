# -*- coding: utf-8 -*-
# @Time    : 3/8/2023 9:56 PM
# @Author  : Gang Qu
# @FileName: pnc_run.py
import os
paradigms = ['rest_pnc', 'emoid_pnc', 'nback_pnc']
cnb_scores = ['wrat', 'pvrt', 'pmat', 'all']
if __name__ == '__main__':
    for paradigm in paradigms:
        for cnb_score in cnb_scores:
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_PNC_multiregression.py --paradigms {p} --max_epochs {e} --cnb_scores {c}'.format(
                p=paradigm, e=int(100), c=cnb_score)
            os.system(cmd)