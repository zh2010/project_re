# -*- coding: utf-8 -*-
import random

import torch

from code_re.MLAC.utils import pos


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
        x, e1, e2, dist1, dist2, e1_pos, e2_pos, y
        :return:
        """
        for batch in self.data:
            batch_size = len(batch)
            X, e1, e2, dist1, dist2, e1_pos, e2_pos, y = list(zip(*batch))

            x_len = max(len(x) for x in X)
            x_ids = torch.LongTensor(batch_size, x_len).fill_(0)
            dist1_padded = torch.LongTensor(batch_size, x_len).fill_(0)
            dist2_padded = torch.LongTensor(batch_size, x_len).fill_(0)
            for i, doc in enumerate(X):
                x_ids[i, :len(doc)] = torch.LongTensor(doc)

                dist1_padded[i, :len(doc)] = torch.LongTensor(dist1[i])
                dist1_padded[i, len(doc):] = torch.LongTensor([pos(e1_pos[i][1] - idx) for idx, _ in enumerate(x_ids[i][len(doc):], start=len(doc))])

                dist2_padded[i, :len(doc)] = torch.LongTensor(dist2[i])
                dist2_padded[i, len(doc):] = torch.LongTensor([pos(e2_pos[i][1] - idx) for idx, _ in enumerate(x_ids[i][len(doc):], start=len(doc))])

            e1_tensor = torch.LongTensor(e1)
            e2_tensor = torch.LongTensor(e2)

            y_tensor = torch.LongTensor(y)

            if self.gpu:
                x_ids = x_ids.pin_memory()
                e1_tensor = e1_tensor.pin_memory()
                e2_tensor = e2_tensor.pin_memory()
                dist1_padded = dist1_padded.pin_memory()
                dist2_padded = dist2_padded.pin_memory()
                y_tensor = y_tensor.pin_memory()

            yield (x_ids, e1_tensor, e2_tensor, dist1_padded, dist2_padded, y_tensor)







