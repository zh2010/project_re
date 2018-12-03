# -*- coding: utf-8 -*-
import os
import re
from collections import defaultdict

from code_re.config import Data_PATH, train_data_path


def find_entity(path):
    sample = []
    for file_name in os.listdir(path):
        with open(os.path.join(path, file_name)) as ftxt:
            lines = "".join([line for line in ftxt])

        entity_dict = defaultdict(dict)
        with open(os.path.join(path, file_name.replace(".txt", ".ann"))) as fann:
            for line in fann:
                if line.startswith("T"):
                    eid, e_des, e_val = line.rstrip().split('\t')
                    e_des = e_des.split(' ')

                    entity_dict[eid] = {
                        "e_name": e_des[0],
                        "s_idx": int(e_des[1]),
                        "e_idx": int(e_des[-1])
                    }
        with open(os.path.join(path, file_name.replace(".txt", ".ann"))) as fann:
            for line in fann:
                if not line.startswith("R"):
                    continue
                rid, r = line.rstrip().split('\t')
                r_name, arg1, arg2 = r.split(' ')
                if entity_dict.get(arg1.split(":")[-1])['e_name'] == r_name.split('_')[0]:
                    e1_info = entity_dict.get(arg1.split(":")[-1])
                    e2_info = entity_dict.get(arg2.split(":")[-1])
                else:
                    e1_info = entity_dict.get(arg2.split(":")[-1])
                    e2_info = entity_dict.get(arg1.split(":")[-1])

                # cut sentence
                min_idx = min(e1_info['s_idx'], e1_info['e_idx'], e2_info['s_idx'], e2_info['e_idx'])
                max_idx = max(e1_info['s_idx'], e1_info['e_idx'], e2_info['s_idx'], e2_info['e_idx'])

                # 确定子句开始位置
                def get_s_idx(newline_cnt, idx):
                    # 有两个\n
                    if newline_cnt > 1:
                        new_line_sub_cnt = 0
                        while idx < min_idx:
                            if lines[idx] == "\n":
                                new_line_sub_cnt += 1
                                # 截取至倒数第三个\n，或是已经到开头
                                if new_line_sub_cnt > 2 or idx == 0:
                                    break
                            idx -= 1
                    # 有一个\n
                    elif newline_cnt > 0:
                        while idx< min_idx:
                            if lines[idx] == "\n" or idx == 0:
                                break
                            idx -= 1
                    else:
                        raise 'min_idx前50字内既没有句号，也没有换行符!! rid: {}'.format(rid)

                    return idx + 1

                # min_idx之前50字内没有句号
                if "。" not in lines[0 if min_idx-50<0 else min_idx-50: min_idx]:
                    newline_cnt = len([c for c in lines[0 if min_idx-50<0 else min_idx-50: min_idx] if c == '\n'])
                    i = min_idx - 1
                    # 有两个\n
                    if newline_cnt > 1:
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
                        raise 'min_idx前50字内既没有句号，也没有换行符!! rid: {}'.format(rid)

                else:
                    i = min_idx - 1
                    while i < min_idx:
                        if lines[i] == "。" and lines[i - 2: i] != "  ":
                            break
                        i -= 1

                # 确定子句终止位置
                if "。" not in lines[max_idx: max_idx+50]:
                    j = max_idx
                    newline_cnt = len([c for c in lines[max_idx: max_idx+50] if c == '\n'])
                    if newline_cnt > 1:
                        new_line_sub_cnt = 0
                        while i >= max_idx:
                            

                else:
                    j = max_idx
                    while j >= max_idx:
                        if lines[j] == "。" and lines[j - 2: j] != "  ":
                            break
                        j += 1

                sub_sent = lines[i + 1: j + 1]
                sub_sent_s_idx = i+1
                sub_sent_e_idx = j+1

                sub_sent_ent = lines[sub_sent_s_idx:e1_info['s_idx']] + "<e1>" + \
                               lines[e1_info['s_idx']: e1_info['e_idx']] + '</e1>' + \
                               lines[e1_info['e_idx']: e2_info['s_idx']] + '<e2>' + \
                               lines[e2_info['s_idx']: e2_info['e_idx']] + '</e2>' + \
                               lines[e2_info['e_idx']:sub_sent_e_idx]
                sub_sent_ent = re.sub('\n', '', sub_sent_ent)

                sample.append((r_name, sub_sent_ent, file_name.split(".")[0], rid))

    with open(os.path.join(Data_PATH, "train.txt"), "w") as fout:
        for r_name, sent, fid, rid in sample:
            if "\t" in sent:
                print("*" * 100)
                print("sent contains Tab !")
            fout.write("{}\t{}\t{}\t{}\n".format(r_name, sent, fid, rid))


if __name__ == '__main__':
    find_entity(train_data_path)
