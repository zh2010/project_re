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


def build_word_dict():
    word_bag = defaultdict(dict)
    get_entity_word(train_data_path, word_bag)
    get_entity_word(test_data_path, word_bag)

    word_list = []
    with open(os.path.join(Data_PATH, "word_dict.txt"), 'w') as f:
        for word in word_bag.keys():
            # 如果实体词中1.有英文; 2.纯中文但词长<5 则准备加入自定义词典
            if not re.search('[\u4e00-\u9fa5]+', word):
                word_list.append([word, word_bag[word]['freq'], word_bag[word]['e_name']])
        word_list.sort(key=lambda x: x[1], reverse=True)

        for ele in word_list:
            f.write('{} {} {}\n'.format(ele[0], ele[1], ele[2]))


def cut_word():
    import jieba
    jieba.load_userdict(os.path.join(Data_PATH, "word_dict.txt"))

    for file_name in ['sample_negative.txt', 'sample_positive.txt']:
        with open(os.path.join(Data_PATH, file_name.replace('.txt', '_cut.txt')), 'w') as fout:
            with open(os.path.join(Data_PATH, file_name)) as f:
                for line in f:
                    r_name, sent, fn, rid = line.rstrip().split('\t')
                    e1_s = sent.index('<e1>')
                    e1_e = sent.index('</e1>')
                    e2_s = sent.index('<e2>')
                    e2_e = sent.index('</e2>')

                    sent_simp = sent.replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', '')

                    sent_cut = jieba.lcut(sent_simp, HMM=False)
                    i = 0
                    offset = []
                    for w in sent_cut:
                        offset.append([i, i+len(w), w])
                        i += len(w)

                    if e1_s < e2_s:

                        for idx, (s_idx, e_idx, w) in enumerate(offset):
                            if e1_s == 0:
                                offset.insert(0, [e1_s, e1_s + 4, '<e1>'])
                                offset = offset[:1] + [[si+4, ei+4, w] for si, ei, w in offset[1:]]
                                break

                            if e_idx == e1_s:
                                offset.insert(idx+1, [e1_s, e1_s + 4, '<e1>'])
                                offset = offset[:idx+2] + [[si+4, ei+4, w] for si, ei, w in offset[idx+2:]]
                                break

                        for idx, (s_idx, e_idx, w) in enumerate(offset):
                            if e_idx == e1_e:
                                offset.insert(idx+1, [e1_e, e1_e + 5, '</e1>'])
                                offset = offset[:idx+2] + [[si+5, ei+5, w] for si, ei, w in offset[idx+2:]]
                                break

                        for idx, (s_idx, e_idx, w) in enumerate(offset):
                            if e_idx == e2_s:
                                offset.insert(idx+1, [e2_s, e2_s + 4, '<e2>'])
                                offset = offset[:idx+2] + [[si+4, ei+4, w] for si, ei, w in offset[idx+2:]]
                                break

                        for idx, (s_idx, e_idx, w) in enumerate(offset):
                            if e_idx == e2_e:
                                offset.insert(idx+1, [e2_e, e2_e + 5, '</e2>'])
                                offset = offset[:idx+2] + [[si+5, ei+5, w] for si, ei, w in offset[idx+2:]]
                                break
                    else:
                        for idx, (s_idx, e_idx, w) in enumerate(offset):
                            if e2_s == 0:
                                offset.insert(0, [e2_s, e2_s + 4, '<e2>'])
                                offset = offset[:1] + [[si+4, ei+4, w] for si, ei, w in offset[1:]]
                                break

                            if e_idx == e2_s:
                                offset.insert(idx+1, [e2_s, e2_s + 4, '<e2>'])
                                offset = offset[:idx+2] + [[si+4, ei+4, w] for si, ei, w in offset[idx+2:]]
                                break

                        for idx, (s_idx, e_idx, w) in enumerate(offset):
                            if e_idx == e2_e:
                                offset.insert(idx+1, [e2_e, e2_e + 5, '</e2>'])
                                offset = offset[:idx+2] + [[si+5, ei+5, w] for si, ei, w in offset[idx+2:]]
                                break

                        for idx, (s_idx, e_idx, w) in enumerate(offset):
                            if e_idx == e1_s:
                                offset.insert(idx+1, [e1_s, e1_s + 4, '<e1>'])
                                offset = offset[:idx+2] + [[si+4, ei+4, w] for si, ei, w in offset[idx+2:]]
                                break

                        for idx, (s_idx, e_idx, w) in enumerate(offset):
                            if e_idx == e1_e:
                                offset.insert(idx+1, [e1_e, e1_e + 5, '</e1>'])
                                offset = offset[:idx+2] + [[si+5, ei+5, w] for si, ei, w in offset[idx+2:]]
                                break

                    sent_cut_upt = [w for _, _, w in offset]
                    fout.write('{}\t{}\t{}\t{}\n'.format(r_name, ' '.join(sent_cut_upt), fn, rid))









def cut_char():
    pass


if __name__ == '__main__':
    cut_word()
