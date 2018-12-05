# -*- coding: utf-8 -*-
import os
import random
import re
from collections import defaultdict, Counter

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from code_re.config import Data_PATH, train_data_path
import numpy as np

np.random.seed(1234)
random.seed(1234)


def split_train_dev():
    with open(os.path.join(Data_PATH, 'sample_negative_cut.txt')) as f:
        negative_list = [line.strip().split('\t') for line in f]
    print('total negative samples size: {}'.format(len(negative_list)))
    negative_list = random.sample(negative_list, 40000)

    with open(os.path.join(Data_PATH, 'sample_positive_cut.txt')) as f:
        positive_list = [line.strip().split('\t') for line in f]
        print('total positive samples size: {}'.format(len(positive_list)))

        # 统计正样本各类比例
        # r_name_list, _, _, _ = zip(*positive_list)
        # for rn, cnt, in Counter(r_name_list).most_common():
        #     print('{}\t{}'.format(rn, cnt))

    total_list = positive_list + negative_list
    total_list = np.random.permutation(total_list)
    r_name_list, sent_cut_list, fns, rids = zip(*total_list)
    X_train, X_test, y_train, y_test = train_test_split(list(zip(sent_cut_list, fns, rids)), r_name_list,
                                                        train_size=0.9,
                                                        random_state=1234, stratify=r_name_list)

    with open(os.path.join(Data_PATH, 'train.txt'), 'w') as f:
        for (sent_cut, fn, rid), y in zip(X_train, y_train):
            f.write('{}\t{}\t{}\t{}\n'.format(y, sent_cut, fn + "_" + rid, 'Train'))

    with open(os.path.join(Data_PATH, 'valid.txt'), 'w') as f:
        for (sent_cut, fn, rid), y in zip(X_test, y_test):
            f.write('{}\t{}\t{}\t{}\n'.format(y, sent_cut, fn + "_" + rid, 'Valid'))


def eda():
    with open(os.path.join(Data_PATH, 'train.txt')) as f:
        train_list = [line.strip().split('\t') for line in f]

    with open(os.path.join(Data_PATH, 'valid.txt')) as f:
        valid_list = [line.strip().split('\t') for line in f]

    total_list = train_list + valid_list

    sent_len_list = []
    rel_dist_list = []
    contains_period_cnt = 0
    contains_two_period_cnt = 0
    contains_three_period_cnt = 0
    for relation, sentence, src, set_type in total_list:
        sentence = sentence.replace('_e11_', 'e11')
        sentence = sentence.replace('_e12_', 'e12')
        sentence = sentence.replace('_e21_', 'e21')
        sentence = sentence.replace('_e22_', 'e22')

        tokens = sentence.split(' ')
        e1 = tokens.index("e12") - 1
        e2 = tokens.index("e22") - 1
        sent_len_list.append(len(tokens))
        rel_dist_list.append(abs(e1 - e2))

        if tokens.count('。') == 1 and tokens[-1] != '。':
            contains_period_cnt += 1
        if tokens.count('。') == 2:
            contains_two_period_cnt += 1
        if tokens.count('。') == 3:
            contains_three_period_cnt += 1

    print('length - max: {}, min: {}, avg: {}, median: {}'.format(max(sent_len_list),
                                                                  min(sent_len_list),
                                                                  np.mean(sent_len_list),
                                                                  np.median(sent_len_list)))

    print('rel_dist - max: {}, min: {}, avg: {}, median: {}'.format(max(rel_dist_list),
                                                                    min(rel_dist_list),
                                                                    np.mean(rel_dist_list),
                                                                    np.median(rel_dist_list)))
    ex1_len_cnt = len([l for l in sent_len_list if l > 1000])
    ex2_len_cnt = len([l for l in sent_len_list if l > 150])
    print('sent len > 1000: {}'.format(ex1_len_cnt))
    print('sent len > 150: {}'.format(ex2_len_cnt))

    print('contains_period_cnt: {}'.format(contains_period_cnt))
    print('contains_two_period_cnt: {}'.format(contains_two_period_cnt))
    print('contains_three_period_cnt: {}'.format(contains_three_period_cnt))


def refine_data_and_labels():
    with open(os.path.join(Data_PATH, 'train.txt')) as f:
        train_list = [line.strip().split('\t') for line in f]

    with open(os.path.join(Data_PATH, 'valid.txt')) as f:
        valid_list = [line.strip().split('\t') for line in f]

    total_list = train_list + valid_list

    data = []
    pos1 = []
    pos2 = []
    max_sentence_length = 0
    for idx in range(0, len(total_list)):
        relation, sentence, src, set_type = total_list[idx]

        sentence = sentence.replace('_e11_', 'e11')
        sentence = sentence.replace('_e12_', 'e12')
        sentence = sentence.replace('_e21_', 'e21')
        sentence = sentence.replace('_e22_', 'e22')

        tokens = sentence.split(' ')
        if max_sentence_length < len(tokens):
            max_sentence_length = len(tokens)

        if len(tokens) > 150:
            continue

        e1 = tokens.index("e12") - 1
        e2 = tokens.index("e22") - 1

        data.append([tokens, e1, e2, relation, src, set_type])

    print("max sentence length = {}\n".format(max_sentence_length))

    with open(os.path.join(Data_PATH, 'data.txt'), 'w') as fout:

        max_sentence_length = 150
        for tokens, e1, e2, relation, src, set_type in tqdm(data):
            p1 = ""
            p2 = ""
            for word_idx in range(len(tokens)):
                p1 += str((max_sentence_length - 1) + word_idx - e1) + " "
                p2 += str((max_sentence_length - 1) + word_idx - e2) + " "

            fout.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(' '.join(tokens), e1, e2, relation, src, set_type, p1, p2))

            # pos1.append(p1)
            # pos2.append(p2)


if __name__ == '__main__':
    split_train_dev()
    refine_data_and_labels()
