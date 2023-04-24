# -*- coding: utf-8 -*-
# @Time    : 2/11/2023 6:31 PM
# @Author  : Gang Qu
# @FileName: main_HCP_regression.py
import argparse
import utils
import os
from os.path import join as pj
from os.path import abspath
import json
import datasets
import torch
from models.load_model import load_model
from train import HCP_regression
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

    save_name = now + config['dataset'] + '_' + config['model_save_suffix'] + '_' + config['model'] + '_' + args.paradigms
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
    if args.dataset == 'dglHCP':
        dataset = datasets.dglHCPdataset(file_name=args.paradigms, pos_enc_dim=config['net_params']['pos_enc_dim'] - 3,
                                         sparse=args.sparse)


    model_save_dir = pj(abspath('results'), save_name + '.pth')
    # model_save_dir = pj(abspath('results/pretrained'), save_name + '.pth')
    # split the dataset and define the dataloader
    train_size = int(args.train_val_test[0] * len(dataset))
    val_size = int(args.train_val_test[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    if args.dataset == 'dglHCP':
        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=args.batch_size,
                                                   collate_fn=datasets.dglHCP_collect_func)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, collate_fn=datasets.dglHCP_collect_func)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                                  collate_fn=datasets.dglHCP_collect_func)
        logger.info('#' * 60)
    # define the model
    model = load_model(config)
    model.to(device)
    logger.info(model)
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
        epoch_loss_train, optimizer = HCP_regression.train_epoch(model, optimizer, device, train_loader, epoch,
                                                                       logger, writer=writer,
                                                                       weight_score=config['net_params']['weight_score'])
        epoch_loss_val, _ = HCP_regression.evaluate_network(model, device, val_loader, epoch, logger, writer=writer,
                                                                  weight_score=config['net_params']['weight_score'])
        epoch_loss_test, _ = HCP_regression.evaluate_network(model, device, test_loader, epoch, logger,
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

                # 'lr_sched': scheduler}

        # if epoch_loss_train < min_test_loss:
        #     min_test_loss = epoch_loss_train
        #     checkpoint = {
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_sched': scheduler}
        early_stopping(epoch_loss_train,  epoch_loss_val)
        if early_stopping.early_stop:
            logger.info("We are at epoch: {}".format(epoch))
            break
    writer.close()
    torch.save(checkpoint, model_save_dir)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='HCP regression')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--max_epochs', default=1, type=int, help='max number of epochs')
    parser.add_argument('--L2', default=1e-6, type=float, help='L2 regularization')
    parser.add_argument('--dropout', default=None, help='dropout rate')
    parser.add_argument('--seed', default=100, type=int, help='random seed')
    parser.add_argument('--config', default='LR_HCPregression_PMAT', help='config file name')
    parser.add_argument('--train_val_test', default=[0.7, 0.1, 0.2], help='train, val, test split')
    parser.add_argument('--dataset', default='dglHCP', help='dataset name')
    parser.add_argument('--sparse', default=30, type=int, help='sparsity for knn graph')
    parser.add_argument('--gl', default=False, help='graph learning beta')
    parser.add_argument('--paradigms', default='rest_hcp',
                        choices=[
                            'social_hcp', 'relational_hcp', 'moto_hcp', 'language_hcp', 'gambling_hcp', 'wm_hcp',
                            'emoid_hcp', 'rest_hcp'
                        ],
                        help='fmri paradigms')
    args = parser.parse_args()

    main(args)