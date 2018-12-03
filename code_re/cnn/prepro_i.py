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
                e1_info = entity_dict.get(arg1.split(":")[-1])
                e2_info = entity_dict.get(arg2.split(":")[-1])

                # cut sentence
                min_idx = min(e1_info['s_idx'], e1_info['e_idx'], e2_info['s_idx'], e2_info['e_idx'])
                max_idx = max(e1_info['s_idx'], e1_info['e_idx'], e2_info['s_idx'], e2_info['e_idx'])

                i = min_idx - 1
                while i < min_idx:
                    if lines[i] == "。" and lines[i - 2: i] != "  ":
                        break
                    i -= 1

                j = max_idx
                while j >= max_idx:
                    if lines[j] == "。" and lines[j - 2: j] != "  ":
                        break
                    j += 1

                sub_sent = lines[i + 1: j + 1]
                sub_sent_ent = sub_sent[:e1_info['s_idx']] + "<e1>" + \
                               sub_sent[e1_info['s_idx']: e1_info['e_idx']] + '</e1>' + \
                               sub_sent[e1_info['e_idx']: e2_info['s_idx']] + '<e2>' + \
                               sub_sent[e2_info['s_idx']: e2_info['e_idx']] + '</e2>' + \
                               sub_sent[e2_info['e_idx']:]
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
