# -*- coding: utf-8 -*-
import os
import re
from collections import defaultdict

from code_re.config import Data_PATH, train_data_path, test_data_path


def get_entity_word(path, word_bag):
    for file_name in os.listdir(path):
        if file_name.endswith('txt'):
            continue
        with open(os.path.join(path, file_name)) as f:
            for line in f:
                if line.startswith('R'):
                    continue
                if any(n in line for n in ['Test_Value', 'Level', 'Operation', 'Reason',
                                           'Frequency', 'Duration', 'Amount', 'SideEff']):
                    continue
                entity_value = line.rstrip().split('\t')[-1].replace(' ', '')

                if entity_value not in word_bag:
                    word_bag[entity_value] = {
                        'freq': 1,
                        'e_name': line.rstrip().split('\t')[1].split(' ')[0]
                    }
                else:
                    word_bag[entity_value]['freq'] += 1


def build_word_bag_1():
    word_bag = defaultdict(dict)
    get_entity_word(train_data_path, word_bag)
    get_entity_word(test_data_path, word_bag)

    word_list = []
    with open(os.path.join(Data_PATH, "word_dict.txt"), 'w') as f:
        for word in word_bag.keys():
            # 如果实体词中有英文则准备加入自定义词典
            if re.search('[^\u4e00-\u9fa5]+', word) or (len(word) < 5 and word_bag[word].get('e_name') not in ['Treatment']):
                word_list.append([word, word_bag[word]['freq'], word_bag[word]['e_name']])
        word_list.sort(key=lambda x: x[1], reverse=True)

        for ele in word_list:
            f.write('{} {} {}\n'.format(ele[0], ele[1], ele[2]))






def build_word_bag_2():
    pass


def cut_char():
    pass


if __name__ == '__main__':
    build_word_bag_1()