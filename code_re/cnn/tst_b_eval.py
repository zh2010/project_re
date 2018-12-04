# -*- coding: utf-8 -*-
import os

from code_re.config import test_b_data_path, Data_PATH


def tst_eval():
    true_list = []
    pred_list = []
    entity_dict = {}
    eda_list = []
    with open(os.path.join(test_b_data_path, '123_11.ann')) as f:
        for line in f:
            if line.startswith('T'):
                continue
            true_list.append(line.strip().split('\t')[-1])

    with open(os.path.join(test_b_data_path, '123_11.ann')) as f:
        for line in f:
            if line.startswith('R'):
                continue
            eid, e_info, _ = line.strip().split('\t')
            entity_dict[eid] = e_info.split(' ')[1]

    with open(os.path.join(Data_PATH, 'submit', '123_11.ann')) as f:
        for line in f:
            str_concat = line.strip().split('\t')[-1]
            pred_list.append(str_concat)
            pred_label, e1_id, e2_id = str_concat.split(' ')
            e1_id = e1_id.split(':')[-1]
            e2_id = e2_id.split(':')[-1]

            eda_list.append()



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