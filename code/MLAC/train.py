# -*- coding: utf-8 -*-
import argparse
import logging
import os
import random
import sys

import torch

from code.MLAC.model import REModel


def lr_decay(optimizer, lr_decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay_rate
    return optimizer


def main():
    args, log = setup()
    if args.resume:
        log.info('[loading previous model ...]')
        map_location = 'cpu' if torch.cuda.is_available() else None
        checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoint.pt'), map_location=map_location)
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        epoch_0 = checkpoint['epoch'] + 1

        model = REModel()

        # sync random seed
        random.setstate(checkpoint['random_state'])
        torch.random.set_rng_state(checkpoint['torch_state'])
        if args.cuda:
            torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])

        if args.reduce_lr:
            lr_decay(model.optimizer, lr_decay_rate=args.reduce_lr)
            log.info('[learning rate reduced by {}]'.format(args.reduce_lr))

        best_val_score = checkpoint['best_eval']

    else:
        model = REModel()
        epoch_0 = 1
        best_val_score = 0.0

    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.info('Epoch {}'.format(epoch))

        # train



def setup():
    parser = argparse.ArgumentParser(description="Train a Relation extraction model")

    # system
    parser.add_argument('--log_per_updates', type=int, default=300)
    parser.add_argument('--model_dir', default="model_dir")
    parser.add_argument('--save_last_only', action='store_true')
    parser.add_argument('--seed', type=int, default=1013)
    parser.add_argument('--cuda', type=str2bool, nargs='?', default=torch.cuda.is_available(), const=True)

    # training
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--resume', default='best_model.pt')
    parser.add_argument('--resume_options', action='store_true')
    parser.add_argument('--reduce_lr', type=float, default=0)
    parser.add_argument('--optimizer', default='sgd',
                        help='supported optimizer: adamax, sgd')
    parser.add_argument('--grad_clipping', type=float, default=5)
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='used for adamax and sgd')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='used for sgd')
    parser.add_argument('--momentum', type=float, default=0,
                        help='used for sgd')
    parser.add_argument('--fix_embeddings', action='store_true')

    # model
    parser.add_argument('--sliding_window', type=int, default=3)
    parser.add_argument('--num_filters', type=int, default=1000)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--max_len', type=int, default=123)

    args = parser.parse_args()

    # set model dir
    os.makedirs(args.model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(args.model_dir)

    if args.resume == "best_model.pt" and not os.path.exists(os.path.join(args.model_dir, args.resume)):
        args.resume = ''

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # set logger
    class ProgressHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            log_entry = self.format(record)
            if record.message.startswith('> '):
                sys.stdout.write('{}\r'.format(log_entry.rstrip()))
                sys.stdout.flush()
            else:
                sys.stdout.write('{}\n'.format(log_entry))

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.model_dir, 'log.txt'))
    fh.setLevel(logging.INFO)
    ch = ProgressHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)

    return args, log


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')