import math
from scipy.stats import spearmanr
import numpy as np
import sys

def test_model(test_data, test_score, We, testing_data):

    def cos_sim(x, y):
        pred1 = np.zeros(dim)
        pred2 = np.zeros(dim)
        for i in x:
            pred1 += We[i]
        for i in y:
            pred2 += We[i]

        p1p2 = np.dot(pred1, pred2)
        p1p2norm = np.sqrt(np.dot(pred1, pred1)) * np.sqrt(np.dot(pred2,pred2))
        return p1p2/p1p2norm  #cosine

    cos = []
    for data in test_data:
        cos.append(cos_sim(data[0], data[1]))

    corr = spearmanr(cos, test_score)
    return corr[0]

def getVectors(file_name):
    vectors = []
    words_vocab = {}  # word:num
    infile = open(file_name, 'r')

    for ind, line in enumerate(infile):
        words = line.strip().split()
        words_vocab[words[0]] = ind
        ''' normalize weight vector '''
        tmp_vec = np.asarray([float(i) for i in words[1:]])
        tmp_vec /= math.sqrt((tmp_vec**2).sum() + 1e-6)
        vectors.append(tmp_vec)

    vectors = np.asarray(vectors).astype('float32')
    return vectors, words_vocab

def getTestingData(file_name, words_vocab):
    infile = open(file_name, 'r')
    test_data, score, testing_data = [], [], []
    for line in infile:
	testing_data.append(line.strip())
        words = line.strip().split('|||')
        tmp = []
        tmp.append([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[0].split()])
        tmp.append([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[1].split()])
        test_data.append(tmp)
        score.append(float(words[2]))
    #print "Number of testing examples: ", len(testing_data)
    return test_data,  score, testing_data

dim = int(sys.argv[2])

We, words_vocab = getVectors(sys.argv[1])

test_data, test_score, testing_data = getTestingData('../data/english/ppdb_test.txt', words_vocab)

print sys.argv[1]
print test_model(test_data, test_score, We, testing_data)

