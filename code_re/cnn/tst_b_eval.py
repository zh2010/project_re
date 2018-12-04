# -*- coding: utf-8 -*-
import os

from code_re.config import test_b_data_path, Data_PATH


def tst_eval():
    true_list = []
    pred_list = []
    with open(os.path.join(test_b_data_path, '123_11.ann')) as f:
        for line in f:
            if line.startswith('T'):
                continue
            true_list.append(line.strip().split('\t')[-1])

    with open(os.path.join(Data_PATH, 'submit', '123_11.ann')) as f:
        for line in f:
            pred_list.append(line.strip().split('\t')[-1])

    right_set = set(true_list).intersection(set(pred_list))
    print('right_set size: {}'.format(len(right_set)))
    print('true_list size: {}'.format(len(true_list)))
    print('pred_list size: {}'.format(len(pred_list)))

    precious = len(right_set) / len(pred_list)
    recall = len(right_set) / len(true_list)
    f1 = precious * recall

    print(precious, recall, f1)


if __name__ == '__main__':
    tst_eval()