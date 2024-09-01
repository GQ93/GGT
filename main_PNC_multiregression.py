# -*- coding: utf-8 -*-
# @Time    : 10/12/2022 11:34 AM
# @Author  : Gang Qu
# @FileName: main_PNC_multiregression.py
import argparse
import utils
import os
from os.path import join as pj
from os.path import abspath
import json
import datasets
import torch
from models.load_model import load_model
from train import PNC_multi_regression
import time
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter


def main(args):
    now = str(time.strftime("%Y_%m_%d_%H_%M"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set random seed for reproducing results
    if args.seed:
        utils.seed_it(args.seed)

    # import config files and update args
    config_file = pj(abspath('configs'), args.config + '.json')
    with open(config_file) as f:
        config = json.load(f)
    config['dataset'] = args.dataset
    if args.dropout:
        config['net_params']['dropout'] = args.dropout

    if args.cnb_scores == 'wrat':
        config['net_params']['weight_score'] = [1.0, 0.0, 0.0]
    elif args.cnb_scores == 'pvrt':
        config['net_params']['weight_score'] = [0.0, 1.0, 0.0]
    elif args.cnb_scores == 'pmat':
        config['net_params']['weight_score'] = [0.0, 0.0, 1.0]
    else:
        config['net_params']['weight_score'] = [0.1, 0.55, 0.35]


    save_name = now + config['dataset'] + '_' + config['model_save_suffix'] + '_' + config['model'] + '_' + args.paradigms + '_' + args.cnb_scores + '_' + str(args.sparse);
    if not os.path.exists(abspath('results')):
        os.makedirs(abspath('results'))
    if not os.path.exists(abspath('results/pretrained')):
        os.makedirs(abspath('results/pretrained'))
    if not os.path.exists(abspath('results/loggers')):
        os.makedirs(abspath('results/loggers'))

    # print the config and args information
    logger = utils.get_logger(name=save_name, path='results/loggers')
    logger.info(args)
    logger.info(config)

    # define tensorboard for visualization of the training
    if not os.path.exists(abspath('results/runs')):
        os.makedirs(abspath('results/runs'))
    writer = SummaryWriter(log_dir=pj(abspath('results/runs'), save_name), flush_secs=30)

    # define dataset
    if args.dataset == 'dglPNC':
        dataset = datasets.dglPNCdataset(file_name=args.paradigms, pos_enc_dim=config['net_params']['pos_enc_dim'] - 3,
                                         sparse=args.sparse, postfix='agethre16')

    print('Total subjects', len(dataset))
    model_save_dir = pj(abspath('results'), save_name + '.pth')
    # model_save_dir = pj(abspath('results/pretrained'), save_name + '.pth')
    # split the dataset and define the dataloader
    train_size = int(args.train_val_test[0] * len(dataset))
    val_size = int(args.train_val_test[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    if args.dataset == 'dglPNC':
        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=args.batch_size,
                                                   collate_fn=datasets.dglPNC_collect_func)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, collate_fn=datasets.dglPNC_collect_func)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                                  collate_fn=datasets.dglPNC_collect_func)
        logger.info('#' * 60)
    # define the model
    model = load_model(config)
    model.to(device)
    logger.info(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total number of parameters: {total_params}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2)
    if config['pretrain']:
        checkpoint = torch.load(abspath(pj('results', config['pretrain_model_name'])))
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    min_test_loss = 1e12

    early_stopping = utils.EarlyStopping(tolerance=20, min_delta=0)
    # train
    for epoch in range(args.max_epochs):
        start_time = time.time()
        epoch_loss_train, optimizer = PNC_multi_regression.train_epoch(model, optimizer, device, train_loader, epoch,
                                                                           logger, writer=writer,
                                                                           weight_score=config['net_params']['weight_score'])
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f'Elapsed time for running the model: {elapsed_time:.4f} seconds')
        epoch_loss_val, _ = PNC_multi_regression.evaluate_network(model, device, val_loader, epoch, logger, writer=writer,
                                                                  weight_score=config['net_params']['weight_score'])
        epoch_loss_test, _ = PNC_multi_regression.evaluate_network(model, device, test_loader, epoch, logger,
                                                                   task='Test', writer=writer,
                                                                   weight_score=config['net_params']['weight_score'])
        scheduler.step(epoch_loss_val)

        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalars(main_tag='Loss/epoch_losses',
                           tag_scalar_dict={'Train': epoch_loss_train,
                                            'Val': epoch_loss_val,
                                            'Test': epoch_loss_test},
                           global_step=epoch)

        scheduler.step(epoch_loss_train)
        epoch_loss_test = epoch_loss_train
        if epoch_loss_test < min_test_loss:
            min_test_loss = epoch_loss_test
            # model_state = copy.deepcopy(model.state_dict())
            # optimizer_state = copy.deepcopy(optimizer.state_dict())
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
                         }

        early_stopping(epoch_loss_train,  epoch_loss_val)
        if early_stopping.early_stop:
            logger.info("We are at epoch: {}".format(epoch))
            break
    writer.close()
    torch.save(checkpoint, model_save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PNC multi-regression')
    parser.add_argument('--lr', default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--max_epochs', default=5, type=int, help='max number of epochs')
    parser.add_argument('--L2', default=1e-6, help='L2 regularization')
    parser.add_argument('--dropout', default=None, help='dropout rate')
    parser.add_argument('--seed', default=100, type=int, help='random seed')
    parser.add_argument('--config', default='GGT_PNCmultiregression', help='config file name')
    parser.add_argument('--train_val_test', default=[0.7, 0.1, 0.2], help='train, val, test split')
    parser.add_argument('--dataset', default='dglPNC', help='dataset name')
    parser.add_argument('--sparse', default=30, type=int, help='sparsity for knn graph')
    parser.add_argument('--gl', default=False, help='graph learning beta')
    parser.add_argument('--cnb_scores', default='wrat',
                        choices=[
                            'wrat', 'pvrt', 'pmat', 'all'
                        ],
                        help='type of cnb scores')
    parser.add_argument('--paradigms', default='emoid_pnc',
                        choices=[
                            'emoid_pnc', 'nback_pnc', 'rest_pnc',
                        ],
                        help='fmri paradigms')
    args = parser.parse_args()

    main(args)


