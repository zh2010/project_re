# -*- coding: utf-8 -*-
import os

from code_re.config import Data_PATH


def train_w2v():
    import gensim

    class ReadData(object):
        def __init__(self):
            pass

        def __iter__(self):
            with open(os.path.join(Data_PATH, 'data.txt')) as f:
                for line in f:
                    sent_cut = line.strip().split('\t')[0]
                    print(sent_cut)
                    tokens = sent_cut.split(' ')
                    yield tokens

    sentences = ReadData()
    model = gensim.models.Word2Vec(sentences, size=150, sg=1, window=6, negative=10, iter=10)

    # 存词向量
    model.wv.save(os.path.join(Data_PATH, 'w2v_1205'))

    print("loss: {}".format(model.get_latest_training_loss()))


def tst_w2v():
    import gensim
    wv_model = gensim.models.KeyedVectors.load(os.path.join(Data_PATH, 'w2v_1205'))

    w = "致"
    for w, s in wv_model.wv.most_similar(positive=[w], topn=20):
        print(w, s)


if __name__ == '__main__':
    tst_w2v()