# -*- coding: utf-8 -*-
import random

import torch


class BatchGen(object):
    def __init__(self, data, batch_size, gpu, evaluation=False):
        self.eval = evaluation
        self.gpu = gpu
        self.batch_size = batch_size

        # sort by len
        data = sorted(data, key=lambda x: len(x[0]))

        # chunk into batches
        data = [data[i: i+batch_size] for i in range(0, len(data), batch_size)]

        # shuffle
        if not evaluation:
            random.shuffle(data)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        """
        x, e1, e2, dist1, dist2
        :return:
        """
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))

            x_len = max(len(x) for x in batch[0])
            x_ids = torch.LongTensor(batch_size, x_len).fill_(0)

