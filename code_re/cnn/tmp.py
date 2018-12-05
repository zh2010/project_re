# -*- coding: utf-8 -*-

if __name__ == '__main__':
    label_dic = {'other': 0,
                 'Test_Disease': 1, 'Symptom_Disease': 2,
                 'Treatment_Disease': 3, 'Drug_Disease': 4,
                 'Anatomy_Disease': 5, 'Frequency_Drug': 6,
                 'Duration_Drug': 7, 'Amount_Drug': 8,
                 'Method_Drug': 9, 'SideEff-Drug': 10}
    new_dic = {}
    cnt = 1
    for idx, (k, v) in enumerate(label_dic.items()):
        print(k)
        if k == 'other':
            new_dic[k] = 0
        else:
            new_dic[k+'(e1,e2)'] = cnt
            cnt += 1
            new_dic[k+'(e2,e1)'] = cnt
            cnt += 1

    print(new_dic)

