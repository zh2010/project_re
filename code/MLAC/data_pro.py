import msgpack
import numpy as np
import logging
from collections import Counter

from code.MLAC.train import setup


logger = logging.getLogger(__name__)


def load_data(file):
    sentences = []
    relations = []
    e1_pos = []
    e2_pos = []

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            line = line.strip().lower().split()
            relations.append(int(line[0]))
            e1_pos.append((int(line[1]), int(line[2])))  # (start_pos, end_pos)
            e2_pos.append((int(line[3]), int(line[4])))  # (start_pos, end_pos)
            sentences.append(line[5:])

    return sentences, relations, e1_pos, e2_pos


def build_dict(sentences):
    word_count = Counter()
    for sent in sentences:
        for w in sent:
            word_count[w] += 1

    ls = word_count.most_common()
    word_dict = {w[0]: index + 2 for (index, w) in enumerate(ls)}
    # leave 0 to PAD and 1 to UNK
    word_dict['<PAD>'] = 0
    word_dict['<UNK>'] = 1
    return word_dict


def load_embedding(emb_file, emb_vocab, word_dict):
    vocab = {}
    with open(emb_vocab, 'r') as f:
        for idx, w in enumerate(f.readlines()):
            w = w.strip().lower()
            vocab[w] = idx

    f = open(emb_file, 'r')
    embed = f.readlines()

    dim = len(embed[0].split())
    num_words = len(word_dict)
    embeddings = np.random.uniform(-0.01, 0.01, size=(num_words, dim))

    pre_trained = 0
    for w in vocab.keys():
        if w in word_dict:
            embeddings[word_dict[w]] = [float(x) for x in embed[vocab[w]].split()]
            pre_trained += 1
    embeddings[0] = np.zeros(dim)

    logger.info(
        'embeddings: %.2f%%(pre_trained) unknown: %d' % (pre_trained / num_words * 100, num_words - pre_trained))

    f.close()
    return embeddings.astype(np.float32)


def pos(x):
    """
    map the relative distance between [0, 123?)
    """
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122


def vectorize(data, word_dict):
    sentences, relations, e1_pos, e2_pos = data

    # replace word with word-id
    e1_vec = []
    e2_vec = []
    sents_vec = []
    # compute relative distance
    dist1 = []
    dist2 = []

    num_data = len(sentences)
    logger.debug('num_data: {}'.format(num_data))

    for idx, (sent, pos1, pos2) in enumerate(zip(sentences, e1_pos, e2_pos)):
        vec = [word_dict[w] if w in word_dict else 0 for w in sent]
        sents_vec.append(vec)

        # last word of e1 and e2
        e1_vec.append(vec[pos1[1]])
        e2_vec.append(vec[pos2[1]])

    for sent, p1, p2 in zip(sents_vec, e1_pos, e2_pos):
        # current word position - last word position of e1 or e2
        dist1.append([pos(p1[1] - idx) for idx, _ in enumerate(sent)])
        dist2.append([pos(p2[1] - idx) for idx, _ in enumerate(sent)])

    return sents_vec, relations, e1_vec, e2_vec, dist1, dist2


def prepare():
    args, _ = setup()
    train_data = load_data("train.txt")
    dev_data = load_data("test.txt")
    word_dict = build_dict(train_data[0] + dev_data[0])

    x, y, e1, e2, dist1, dist2 = vectorize(train_data, word_dict, args.max_len)
    y = np.array(y).astype(np.int64)
    np_cat = np.concatenate(
        (x, np.array(e1).reshape(-1, 1), np.array(e2).reshape(-1, 1), np.array(dist1), np.array(dist2)), 1)

    e_x, e_y, e_e1, e_e2, e_dist1, e_dist2 = vectorize(dev_data, word_dict, args.max_len)
    e_y = np.array(e_y).astype(np.int64)
    eval_cat = np.concatenate(
        (e_x, np.array(e_e1).reshape(-1, 1), np.array(e_e2).reshape(-1, 1), np.array(e_dist1), np.array(e_dist2)), 1)

    embed_file = 'embeddings.txt'
    vac_file = 'words.lst'
    embedding = load_embedding(embed_file, vac_file, word_dict)

    # save
    meta = {
        "embeddings": embedding.tolist()
    }

    result = {
        "train_x": np_cat.tolist(),
        "train_y": y.tolist(),
        "eval_x": eval_cat.tolist(),
        "eval_y": e_y.tolist(),
    }

    with open("meta.msgpack", "wb") as f:
        msgpack.dump(meta, f)

    with open("data.msgpack", "wb") as f:
        msgpack.dump(result, f)


if __name__ == '__main__':
    prepare()






