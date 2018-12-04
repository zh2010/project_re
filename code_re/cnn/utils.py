import numpy as np


class2label = {'other': 0,
               'Test_Disease': 1, 'Symptom_Disease': 2,
               'Treatment_Disease': 3, 'Drug_Disease': 4,
               'Anatomy_Disease': 5, 'Frequency_Drug': 6,
               'Duration_Drug': 7, 'Amount_Drug': 8,
               'Method_Drug': 9, 'SideEff-Drug': 10}

label2class = {v: k for k, v in class2label.items()}


def load_word2vec(embedding_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) / np.sqrt(len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load word2vec file {0}".format(embedding_path))
    cnt = 0
    with open(embedding_path, "rb") as f:
        header = f.readline()
        vocab_size, layer_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            idx = vocab.vocabulary_.get(word)
            if idx != 0:
                cnt += 1
                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    print("in vocab cnt: {}".format(cnt))
    return initW
