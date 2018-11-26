# -*- coding: utf-8 -*-
import argparse
import logging
import os
import random
import sys
from datetime import datetime

import msgpack
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support

from code_re.MLAC.generate_batch import BatchGen
from code_re.MLAC.model import REModel


def lr_decay(optimizer, lr_decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay_rate
    return optimizer


def load_data(opt):
    with open("meta.msgpack", 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')

    with open("data.msgpack", 'rb') as f:
        data = msgpack.load(f, encoding='utf8')

    embedding = torch.Tensor(meta["embeddings"])
    train = data["train"]
    valid = data["valid"]

    opt["pretrained_words"] = True
    opt["vocab_size"] = embedding.size(0)
    opt["embedding_dim"] = embedding.size(1)
    opt["num_positions"] = len(set([d for l in train + valid for d in l[3] + l[4]]))
    yl = sorted(set([l[-1] for l in train + valid]))
    opt["num_relations"] = len(yl)
    opt["relation_name"] = yl

    return train, valid, embedding, opt


def main():
    args, log = setup()
    log.info('[Program starts. Loading data ...]')
    train, valid, embedding, opt = load_data(vars(args))
    log.info('[Data loaded]')

    if args.resume:
        log.info('[loading previous model ...]')
        map_location = 'cpu' if torch.cuda.is_available() else None
        checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoint.pt'), map_location=map_location)
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        epoch_0 = checkpoint['epoch'] + 1

        model = REModel(opt, embedding, state_dict)

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
        model = REModel(opt, embedding)
        epoch_0 = 1
        best_val_score = 0.0

    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.info('Epoch {}'.format(epoch))

        # train
        batches = BatchGen(train, batch_size=args.batch_size, gpu=args.cuda)
        start = datetime.now()
        for i, batch in enumerate(batches):
            model.update(batch)

            # evaluate batch
            if i % args.log_per_updates == 0:
                log.info('> epoch [{0:2}] updates[{1:6}] train loss[{2:5f}] remaining[{3}]'.format(
                    epoch, model.updates, model.loss,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]
                ))
                true_y = batch[-1]
                log.info(classification_report(true_y, model.pred.cpu()))
                p, r, f1, s = precision_recall_fscore_support(true_y, model.pred.cpu(), average='micro')
        log.info('\n')

        # save
        model_file = os.path.join(args.model_dir, "checkpoint.pt")
        model.save(epoch, [p, r, f1, best_val_score], model_file)

        # evaluate
        true_y_list, pred_y_list = [], []
        batches = BatchGen(valid, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
        for i, batch in enumerate(batches):
            output = model.predict(batch)
            true_y = batch[-1]
            true_y_list.extend(true_y)
            pred_y_list.extend(output.cpu())

        log.info(classification_report(true_y_list, pred_y_list))

        pv, rv, f1v, sv = precision_recall_fscore_support(true_y_list, pred_y_list, average='micro')

        log.info('epoch {} dev f1: {}, best f1: {}'.format(epoch, f1v, best_val_score))

        if f1v > best_val_score:
            best_val_score = f1v
            model_file = os.path.join(args.model_dir, "best_model.pt")
            model.save(epoch, [pv, rv, f1v, best_val_score], model_file)


def setup():
    parser = argparse.ArgumentParser(description="Train a Relation extraction model")

    # system
    parser.add_argument('--log_per_updates', type=int, default=300)
    parser.add_argument('--model_dir', default="model_dir")
    parser.add_argument('--save_last_only', action='store_true')
    parser.add_argument('--seed', type=int, default=1013)
    parser.add_argument('--cuda', type=str2bool, nargs='?', default=torch.cuda.is_available(), const=True)

    # training
    parser.add_argument('--epochs', type=int, default=100)
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
    parser.add_argument('--fix_embedding', action='store_true')

    # model
    parser.add_argument('--sliding_window', type=int, default=3)
    parser.add_argument('--num_filters', type=int, default=1000)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--max_len', type=int, default=123)
    parser.add_argument('--position_embedding_dim', type=int, default=25)

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


if __name__ == '__main__':
    main()

