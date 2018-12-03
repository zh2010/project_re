# -*- coding: utf-8 -*-
import os
import re
from collections import defaultdict

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
                    eid, e_des, e_val = line.rstrip().split('\t')
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

        period_idx_list = [idx for idx, c in enumerate(lines) if c == "。"]
        period_idx_list.insert(0, 0)
        for idx, pid in enumerate(period_idx_list):
            if idx == len(period_idx_list) - 1:
                break
            next_peroid_id = period_idx_list[idx + 1]
            sub_sent = lines[pid+1: next_peroid_id+1]
            if len(sub_sent) > 150:
                continue
            sub_sent_s_idx = pid + 1
            sub_sent_e_idx = next_peroid_id + 1

            tmp_entity_bag = []
            for eid in entity_dict.keys():
                if entity_dict[eid].get('s_idx') >= sub_sent_s_idx and entity_dict[eid].get('e_idx') <= sub_sent_e_idx:
                    tmp_entity_bag.append(eid)

            if len(tmp_entity_bag) > 1:
                # 列表元素两两配对
                matched_entity_dict = set()
                for e1_id in tmp_entity_bag:
                    for e2_id in tmp_entity_bag:
                        if e1_id != e2_id \
                                and e1_id + "_" + e2_id not in relation_dict \
                                and e2_id + "_" + e1_id not in relation_dict \
                                and e1_id + "_" + e2_id not in matched_entity_dict \
                                and e2_id + "_" + e1_id not in matched_entity_dict:
                            e1_info = entity_dict.get(e1_id)
                            e2_info = entity_dict.get(e2_id)

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

                            matched_entity_dict.add(e1_id + "_" + e2_id)
                            matched_entity_dict.add(e2_id + "_" + e1_id)

    with open(os.path.join(Data_PATH, "sample_negative.txt"), "w") as fout:
        for r_name, sent, fid, rid in sample:
            if "\t" in sent:
                print("*" * 100)
                print("sent contains Tab !")
            fout.write("{}\t{}\t{}\t{}\n".format(r_name, sent, fid, rid))


if __name__ == '__main__':
    find_entity(train_data_path)



