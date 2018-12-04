# -*- coding: utf-8 -*-
import os
import re
from collections import defaultdict

from tqdm import tqdm

from code_re.config import Data_PATH, train_data_path


def find_entity(path):
    sample = []
    for file_name in os.listdir(path):
        if file_name.endswith('ann'):
            continue

        entity_dict = defaultdict(dict)
        with open(os.path.join(path, file_name.replace(".txt", ".ann"))) as fann:
            for line in fann:
                if line.startswith("T"):
                    eid, e_des, e_val = line.strip().split('\t')
                    e_des = e_des.split(' ')

                    entity_dict[eid] = {
                        "e_name": e_des[0],
                        "s_idx": int(e_des[1]),
                        "e_idx": int(e_des[-1])
                    }

        relation_dict = set()
        with open(os.path.join(path, file_name.replace(".txt", ".ann"))) as fann:
            for line in fann:
                if not line.startswith("R"):
                    continue
                rid, r = line.rstrip().split('\t')
                r_name, arg1, arg2 = r.split(' ')

                relation_dict.add(arg1.split(":")[-1] + "_" + arg2.split(":")[-1])
                relation_dict.add(arg2.split(":")[-1] + "_" + arg1.split(":")[-1])

        with open(os.path.join(path, file_name)) as ftxt:
            lines = "".join([line for line in ftxt])

        sorted_entity_dict = sorted(entity_dict.items(), key=lambda x: x[1].get('e_idx'))
        matched_set = set()
        label_dic = {'other': 0,
                     'Test_Disease': 1, 'Symptom_Disease': 2,
                     'Treatment_Disease': 3, 'Drug_Disease': 4,
                     'Anatomy_Disease': 5, 'Frequency_Drug': 6,
                     'Duration_Drug': 7, 'Amount_Drug': 8,
                     'Method_Drug': 9, 'SideEff-Drug': 10}

        for e1_id_, e1_info_ in tqdm(sorted_entity_dict):
            nearby_entity_dict = [(eid_, einfo_) for eid_, einfo_ in sorted_entity_dict if 0 < einfo_['e_idx'] - e1_info_['e_idx'] <= 100]
            for e2_id_, e2_info_ in nearby_entity_dict:
                if e1_info_['e_name'] == e2_info_['e_name']:
                    continue

                if e1_info_['e_name'] + "_" + e2_info_['e_name'] not in label_dic and \
                                                e2_info_['e_name'] + "_" + e1_info_['e_name'] not in label_dic and \
                                                e1_info_['e_name'] + "-" + e2_info_['e_name'] not in label_dic and \
                                                e2_info_['e_name'] + "-" + e1_info_['e_name'] not in label_dic:
                    continue

                if e1_id_ + '_' + e2_id_ in matched_set or e2_id_ + '_' + e1_id_ in matched_set:
                    continue

                if e1_id_ + '_' + e2_id_ in relation_dict or e2_id_ + '_' + e1_id_ in relation_dict:
                    continue

                # entity dist 须>100
                # entity_name pair必须在关系列表
                # entity_id pair已配对则pass
                # entity_id pair不能在关系列表

                if e2_info_['e_name'] + "_" + e1_info_['e_name'] in label_dic or \
                                                e2_info_['e_name'] + "-" + e1_info_['e_name'] in label_dic:
                    e1_info, e2_info = e2_info_, e1_info_
                    e1_id, e2_id = e2_id_, e1_id_
                else:
                    e1_info, e2_info = e1_info_, e2_info_
                    e1_id, e2_id = e1_id_, e2_id_

                min_idx = min(e1_info['s_idx'], e1_info['e_idx'], e2_info['s_idx'], e2_info['e_idx'])
                max_idx = max(e1_info['s_idx'], e1_info['e_idx'], e2_info['s_idx'], e2_info['e_idx'])

                # 确定子句开始位置
                if "。" not in lines[0 if min_idx - 70 < 0 else min_idx - 70: min_idx]:
                    # min_idx之前70字内没有句号
                    newline_cnt = len(
                        [c for c in lines[0 if min_idx - 70 < 0 else min_idx - 70: min_idx] if c == '\n'])
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
                if "。" not in lines[max_idx: max_idx + 70]:
                    j = max_idx
                    newline_cnt = len([c for c in lines[max_idx: max_idx + 70] if c == '\n'])
                    if newline_cnt > 2:
                        new_line_sub_cnt = 0
                        while j >= max_idx:
                            if lines[j + 1] == '\n':
                                new_line_sub_cnt += 1
                                if new_line_sub_cnt > 2 or j == len(lines) - 1:
                                    break
                            j += 1
                    elif newline_cnt > 0:
                        while j >= max_idx:
                            if lines[j + 1] == "\n" or j == len(lines) - 1:
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
                sub_sent_s_idx = i + 1
                sub_sent_e_idx = j + 1

                if e1_info['s_idx'] < e2_info['s_idx']:
                    sub_sent_ent = lines[sub_sent_s_idx:e1_info['s_idx']] + "<e1>" + \
                                   lines[e1_info['s_idx']: e1_info['e_idx']] + '</e1>' + \
                                   lines[e1_info['e_idx']: e2_info['s_idx']] + '<e2>' + \
                                   lines[e2_info['s_idx']: e2_info['e_idx']] + '</e2>' + \
                                   lines[e2_info['e_idx']:sub_sent_e_idx]
                else:
                    sub_sent_ent = lines[sub_sent_s_idx:e2_info['s_idx']] + "<e2>" + \
                                   lines[e2_info['s_idx']: e2_info['e_idx']] + '</e2>' + \
                                   lines[e2_info['e_idx']: e1_info['s_idx']] + '<e1>' + \
                                   lines[e1_info['s_idx']: e1_info['e_idx']] + '</e1>' + \
                                   lines[e1_info['e_idx']:sub_sent_e_idx]
                sub_sent_ent = re.sub('[\n\s]', '', sub_sent_ent)

                sample.append(('other', sub_sent_ent, file_name.split(".")[0], 'R-other'))

                matched_set.add(e1_id + "_" + e2_id)
                matched_set.add(e2_id + "_" + e1_id)

    with open(os.path.join(Data_PATH, "sample_negative.txt"), "w") as fout:
        for r_name, sent, fid, rid in sample:
            if "\t" in sent:
                print("*" * 100)
                print("sent contains Tab !")
            fout.write("{}\t{}\t{}\t{}\n".format(r_name, sent, fid, rid))


if __name__ == '__main__':
    find_entity(train_data_path)
