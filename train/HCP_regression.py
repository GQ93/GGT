# -*- coding: utf-8 -*-
# @Time    : 2/11/2023 6:27 PM
# @Author  : Gang Qu
# @FileName: HCP_regression.py
import torch
import torch.nn as nn
import dgl
import numpy as np
import utils
from tqdm import tqdm


def train_epoch(model, optimizer, device, data_loader, epoch, logger=None, writer=None, weight_score=None):
    model.train()
    epoch_loss = 0
    predicted_scores = []
    target_scores = []
    if logger:
        logger.info('-'*60)
        logger.info('EPOCH:{} (lr={})'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    else:
        print('-'*60)
        print('EPOCH:{} (lr={})'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    for iter, sample_i in enumerate(tqdm(data_loader, desc="Training iterations!")):
        # sample = {'g': g, 'y_score': y_score, 'y_age': y_age, 'mni': self.mni, 'mni_dist': self.mni_dist, 'roi_index': self.roi_index}
        batch_graphs = sample_i['g']
        batch_targets = sample_i['y_score']

        batch_pos_enc = batch_graphs.ndata['p'].to(device)
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['x'].to(device)  # num x feat
        batch_e = batch_graphs.edata['x'].to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()

        batch_scores, _ = model(h=batch_x, e=batch_e, p=batch_pos_enc, g=batch_graphs)
        loss = model.loss(batch_scores, batch_targets, weight_score, experiment='HCP')
        loss.backward()

        optimizer.step()
        epoch_loss += loss.detach().item()

        predicted_scores.append(batch_scores.cpu().detach().numpy())
        target_scores.append(batch_targets.cpu().detach().numpy())

        # if logger:
        #     logger.info('Iteration {} loss={}'.format(iter, loss.detach().item()))
        # else:
        #     print('Iteration {} loss={}'.format(iter, loss.detach().item()))
    epoch_loss /= (iter + 1)
    predicted_scores = np.concatenate(predicted_scores, axis=0)
    target_scores = np.concatenate(target_scores, axis=0)
    target_scores = target_scores.reshape(-1, 1)
    rmse = utils.evaluate_mat(predicted_scores, target_scores, method='RMSE')
    mae = utils.evaluate_mat(predicted_scores, target_scores, method='MAE')
    if writer:
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('PMAT/RMSE_'+'Train', rmse[0], epoch)
        writer.add_scalar('PMAT/MAE_'+'Train', mae[0], epoch)

    if logger:
        logger.info('loss={} RMSE={} MAE={}'.format(epoch_loss, rmse, mae))
    else:
        print('loss={} RMSE={} MAE={}'.format(epoch_loss, rmse, mae))
    logger.info('-' * 60)
    return epoch_loss, optimizer


def evaluate_network(model, device, data_loader, epoch, logger=None, task='Val',  writer=None, weight_score=None):
    model.eval()
    epoch_test_loss = 0
    predicted_scores = []
    target_scores = []
    attention = []
    with torch.no_grad():
        for iter, sample_i in enumerate(tqdm(data_loader, desc=task + " iterations!")):
            batch_graphs = sample_i['g']
            batch_targets = sample_i['y_score']

            batch_pos_enc = batch_graphs.ndata['p'].to(device)
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['x'].to(device)  # num x feat
            batch_e = batch_graphs.edata['x'].to(device)
            batch_targets = batch_targets.to(device)

            batch_scores, gs = model(h=batch_x, e=batch_e, p=batch_pos_enc, g=batch_graphs)

            for g in gs:
                attention_score = g.edata['alpha_ij']
                pos_vec = g.ndata['p']
                attention.append((attention_score, pos_vec, g.edges()))
            loss = model.loss(batch_scores, batch_targets, weight_score, experiment='HCP')
            epoch_test_loss += loss.detach().item()
            predicted_scores.append(batch_scores.cpu().detach().numpy())
            target_scores.append(batch_targets.cpu().detach().numpy())
        epoch_test_loss /= (iter + 1)
    predicted_scores = np.concatenate(predicted_scores, axis=0)
    target_scores = np.concatenate(target_scores, axis=0)
    target_scores = target_scores.reshape(-1, 1)
    rmse = utils.evaluate_mat(predicted_scores, target_scores, method='RMSE')
    mae = utils.evaluate_mat(predicted_scores, target_scores, method='MAE')
    if writer:
        writer.add_scalar('Loss/'+task, epoch_test_loss, epoch)
        writer.add_scalar('PMAT/RMSE_'+task, rmse, epoch)
        writer.add_scalar('PMAT/MAE_'+task, mae, epoch)
    if logger:
        logger.info('{}_loss={} RMSE={} MAE={}'.format(task, epoch_test_loss, rmse, mae))
    else:
        print('{}_loss={} RMSE={} MAE={}'.format(task, epoch_test_loss, rmse, mae))
    return epoch_test_loss, attention
