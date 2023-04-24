# -*- coding: utf-8 -*-
# @Time    : 3/26/2023 7:45 PM
# @Author  : Gang Qu
# @FileName: ablation_study.py
import os
if __name__ == '__main__':
    paradigms = ['rest']
    models = ['GGT']
    # for paradigm in paradigms:
    #     for model in models:
    #         cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_HCP_regression.py --paradigms {p}_hcp --max_epochs {e} --config {m}_HCPregression_PMAT_noRWMNI'.format(
    #             p=paradigm, e=int(100), m=model)
    #         os.system(cmd)
    #
    #
    # for paradigm in paradigms:
    #     for model in models:
    #         cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_HCP_regression.py --paradigms {p}_hcp --max_epochs {e} --config {m}_HCPregression_PMAT_noMNI'.format(
    #             p=paradigm, e=int(100), m=model)
    #         os.system(cmd)

    # for paradigm in paradigms:
    #     for model in models:
    #         cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_HCP_regression.py --paradigms {p}_hcp --max_epochs {e} --config {m}_HCPregression_PMAT_noRW'.format(
    #             p=paradigm, e=int(100), m=model)
    #         os.system(cmd)


    for paradigm in paradigms:
        for model in models:
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_PNC_multiregression.py --paradigms {p}_pnc --max_epochs {e} --cnb_scores all --config {m}_PNCmultiregression_noRWMNI'.format(
                p=paradigm, e=int(100), m=model)
            os.system(cmd)

    for paradigm in paradigms:
        for model in models:
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_PNC_multiregression.py --paradigms {p}_pnc --max_epochs {e} --cnb_scores all --config {m}_PNCmultiregression_noMNI'.format(
                p=paradigm, e=int(100), m=model)
            os.system(cmd)

    for paradigm in paradigms:
        for model in models:
            cmd = 'set CUDA_VISIBLE_DEVICES=0 &python main_PNC_multiregression.py --paradigms {p}_pnc --max_epochs {e} --cnb_scores all --config {m}_PNCmultiregression_noRW'.format(
                p=paradigm, e=int(100), m=model)
            os.system(cmd)
