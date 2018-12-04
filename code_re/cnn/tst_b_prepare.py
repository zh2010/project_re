# -*- coding: utf-8 -*-
import os
import re
from collections import defaultdict

import jieba

from code_re.config import Data_PATH, test_b_data_path

jieba.load_userdict(os.path.join(Data_PATH, "word_dict.txt"))
jieba.add_word('_e11_')
jieba.add_word('_e12_')
jieba.add_word('_e21_')
jieba.add_word('_e22_')


def prepare_tst():
    # 1.测试样本：每个测试文件读进来，结尾符相隔100字符以内的实体都配对，顺序按照关系名称
    # 2.分词
    # 3.准备模型所需格式
    # 4.预测
    for file_name in os.listdir(test_b_data_path):
        if file_name.endswith('ann'):
            continue
        with open(os.path.join(test_b_data_path, file_name)) as ftxt:
            lines = "".join([line for line in ftxt])

        entity_dict = defaultdict(dict)
        with open(os.path.join(test_b_data_path, file_name.replace('txt', 'ann'))) as fann:
            for line in fann:
                if line.startswith("T"):
                    eid, e_des, e_val = line.rstrip().split('\t')
                    e_des = e_des.split(' ')

                    entity_dict[eid] = {
                        "e_name": e_des[0],
                        "s_idx": int(e_des[1]),
                        "e_idx": int(e_des[-1])
                    }
        sorted_entity_dict = sorted(entity_dict.items(), key=lambda x: x[1].get('e_idx'))
        matched_set = set()
        rel_dic = {'other': 0,
                   'Test_Disease': 1, 'Symptom_Disease': 2,
                   'Treatment_Disease': 3, 'Drug_Disease': 4,
                   'Anatomy_Disease': 5, 'Frequency_Drug': 6,
                   'Duration_Drug': 7, 'Amount_Drug': 8,
                   'Method_Drug': 9, 'SideEff-Drug': 10}

        max_sentence_length = 0
        data_list = []
        for e1_id, e1_info in sorted_entity_dict:
            for e2_id, e2_info in sorted_entity_dict:
                if abs(e1_info.get('e_idx') - e2_info.get('e_idx')) <= 30 and \
                        (e1_info['e_name'] + "_" + e2_info['e_name'] in rel_dic or
                         e2_info['e_name'] + "_" + e1_info['e_name'] in rel_dic or
                         e1_info['e_name'] + "-" + e2_info['e_name'] in rel_dic or
                         e2_info['e_name'] + "-" + e1_info['e_name'] in rel_dic
                        ) and \
                        (e1_id + '_' + e2_id not in matched_set and e2_id + '_' + e1_id not in matched_set):

                    if e2_info['e_name'] + "_" + e1_info['e_name'] in rel_dic or e2_info['e_name'] + "-" + e1_info['e_name'] in rel_dic:
                        e1_info, e2_info = e2_info, e1_info
                        e1_id, e2_id = e2_id, e1_id

                    min_idx = min(e1_info['s_idx'], e1_info['e_idx'], e2_info['s_idx'], e2_info['e_idx'])
                    max_idx = max(e1_info['s_idx'], e1_info['e_idx'], e2_info['s_idx'], e2_info['e_idx'])

                    # 确定子句开始位置
                    if "。" not in lines[0 if min_idx-70<0 else min_idx-70: min_idx]:
                        # min_idx之前70字内没有句号
                        newline_cnt = len([c for c in lines[0 if min_idx-70<0 else min_idx-70: min_idx] if c == '\n'])
                        i = min_idx - 1
                        # 有两个\n
                        if newline_cnt > 2:
                            new_line_sub_cnt = 0
                            while i < min_idx:
                                if lines[i] == "\n":
                                    new_line_sub_cnt += 1
                                    # 截取至倒数第三个\n，或是已经到开头
                                    if new_line_sub_cnt > 2 or i == 0:
                                        break
                                i -= 1
                        # 有一个\n
                        elif newline_cnt > 0:
                            while i < min_idx:
                                if lines[i] == "\n" or i == 0:
                                    break
                                i -= 1
                        else:
                            print('=' * 50)
                            print('min_idx前70字内既没有句号，也没有换行符!! file: {} line: {}'.format(file_name, line))
                            i = min_idx - 1

                    else:
                        i = min_idx - 1
                        while i < min_idx:
                            if lines[i] == "。" and lines[i - 2: i] != "  ":
                                break
                            i -= 1

                    # 确定子句终止位置
                    if "。" not in lines[max_idx: max_idx+70]:
                        j = max_idx
                        newline_cnt = len([c for c in lines[max_idx: max_idx+70] if c == '\n'])
                        if newline_cnt > 2:
                            new_line_sub_cnt = 0
                            while j >= max_idx:
                                if lines[j+1] == '\n':
                                    new_line_sub_cnt += 1
                                    if new_line_sub_cnt > 2 or j == len(lines)-1:
                                        break
                                j += 1
                        elif newline_cnt > 0:
                            while j >= max_idx:
                                if lines[j+1] == "\n" or j == len(lines) - 1:
                                    break
                                j += 1
                        else:
                            print('*' * 100)
                            print('max_idx后70字内没有句号，也没有换行符. file: {} line: {}'.format(file_name, line))
                            j = max_idx

                    else:
                        j = max_idx
                        while j >= max_idx:
                            if lines[j] == "。" and lines[j - 2: j] != "  ":
                                break
                            j += 1

                    sub_sent = lines[i + 1: j + 1]
                    sub_sent_s_idx = i+1
                    sub_sent_e_idx = j+1

                    if e1_info['s_idx'] < e2_info['s_idx']:
                        sub_sent_ent = lines[sub_sent_s_idx:e1_info['s_idx']] + "_e11_" + \
                                       lines[e1_info['s_idx']: e1_info['e_idx']] + '_e12_' + \
                                       lines[e1_info['e_idx']: e2_info['s_idx']] + '_e21_' + \
                                       lines[e2_info['s_idx']: e2_info['e_idx']] + '_e22_' + \
                                       lines[e2_info['e_idx']:sub_sent_e_idx]
                    else:
                        sub_sent_ent = lines[sub_sent_s_idx:e2_info['s_idx']] + "_e21_" + \
                                       lines[e2_info['s_idx']: e2_info['e_idx']] + '_e22_' + \
                                       lines[e2_info['e_idx']: e1_info['s_idx']] + '_e11_' + \
                                       lines[e1_info['s_idx']: e1_info['e_idx']] + '_e12_' + \
                                       lines[e1_info['e_idx']:sub_sent_e_idx]

                    sub_sent_ent = re.sub('[\n\s]', '', sub_sent_ent)

                    # tokenize
                    sent_cut = jieba.lcut(sub_sent_ent, HMM=False)
                    sentence = ' '.join(sent_cut)

                    # get entity distance
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

                    p1 = ""
                    p2 = ""
                    for word_idx in range(len(tokens)):
                        p1 += str((max_sentence_length - 1) + word_idx - e1) + " "
                        p2 += str((max_sentence_length - 1) + word_idx - e2) + " "

                    data_list.append([' '.join(tokens), e1, e2, p1, p2, e1_id, e2_id])

                    matched_set.add(e1_id + '_' + e2_id)

        with open(os.path.join(Data_PATH, 'tmp', file_name.replace('txt', 'sample')), 'w') as f:
            for data in data_list:
                f.write('\t'.join(list(map(str, data))) + '\n')


if __name__ == '__main__':
    prepare_tst()

