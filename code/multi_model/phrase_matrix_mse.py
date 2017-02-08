from copy import copy

import numpy as np
import theano
import theano.tensor as  T
from collections import OrderedDict
import argparse
import math
import random
from scipy.stats import spearmanr

class Layer(object):
    def __init__(self, We_init, W_init, activation, LW, LL):
        self.activation = activation

        self.We = theano.shared(value=copy(We_init), name='We', borrow=True)
        self.W = theano.shared(value=copy(W_init), name='W', borrow=True)

        self.params = [self.W,  self.We]

        self.L2_1 = T.sum((self.W - W_init) ** 2)
        self.L2_qrt = T.sum(self.W**2)

        self.LW = LW
        self.LL = LL

        self.word_reg = 0.5 * self.L2_1 * self.LW
        self.word_reg += 0.5 * self.L2_qrt * self.LL

    # input_x is batch of indice of training data, which has different length
    def output(self, input_x, We_mask):

        # B * N * E | BN * E
        res1 = (self.We*We_mask)[input_x.flatten()].reshape([input_x.shape[0] *
                                                            input_x.shape[1],
                                                            self.We.shape[1]])
        # E * E
        res2 = self.W

        # BN * E
        lin_output = self.activation(T.dot(res1, res2).reshape([input_x.shape[0], input_x.shape[1], -1])).sum(1)
        return lin_output

def squared_error(mlp, x1, x2, We_mask):
    return T.sum((mlp.output(x1, We_mask, ) - mlp.output(x2, We_mask)) ** 2)

def adagrad(loss, params, learning_rate=0.05, epsilon=1e-6):
    grads = T.grad(loss, params)
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype='float32'),broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        updates[param] = T.cast(param - (learning_rate * grad / T.sqrt(accu_new + epsilon)), 'float32')
    return updates

def test_model(mlp_output, test_data_x, test_data_y, test_score, We_mask_init):
    # cos = []

    predication1 = mlp_output(test_data_x,  We_mask_init)
    predication2 = mlp_output(test_data_y, We_mask_init)

    # for i in range(len(predication1)):
    #     cos.append(1 - spatial.distance.cosine(predication1[i].tolist(), predication2[i].tolist())) #cosine simialrity for corresponding line

    p1p2 = (predication1*predication2).sum(axis=1) #B
    p1p2norm = np.sqrt((predication1 ** 2).sum(axis = 1)) * np.sqrt((predication2 ** 2).sum(axis=1))
    cos = p1p2/p1p2norm  #cosine

    corr = spearmanr(cos, test_score)
    return corr

def valid_model(mlp_output, valid_data_x, valid_data_y, valid_score, We_mask_init):

    predication1 = mlp_output(valid_data_x, We_mask_init) #B*E
    predication2 = mlp_output(valid_data_y, We_mask_init)

    p1p2 = (predication1*predication2).sum(axis=1) #B
    p1p2norm = np.sqrt((predication1 ** 2).sum(axis = 1)) * np.sqrt((predication2 ** 2).sum(axis=1))
    cos = p1p2/p1p2norm  #cosine

    corr = spearmanr(cos, valid_score)
    return corr

def getVectors(file_name, params):
    vectors = []
    words_vocab = {}  # word:num
    infile = open(file_name, 'r')
    vectors.append([0 for i in range(params.dim)])  #first one for other words
    for ind, line in enumerate(infile):
        words = line.strip().split()
        words_vocab[words[0]] = ind+1
        ''' normalize weight vector '''
        tmp_vec = np.asarray([float(i) for i in words[1:]])
        tmp_vec /= math.sqrt((tmp_vec**2).sum() + 1e-6)
        vectors.append(tmp_vec)

    vectors = np.asarray(vectors).astype('float32')
    return vectors, words_vocab

def getTrainingData(file_name, words_vocab):
    infile = open(file_name, 'r')
    training_data = []
    for line in infile:
        words = line.strip().split('|||')
        tmp = []
        tmp.append([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[0].split()])
        tmp.append([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[1].split()])
        training_data.append(tmp)

    #print "Number of training examples: ", len(training_data)
    return training_data

def getValidData(file_name, words_vocab):
    infile = open(file_name, 'r')
    valid_data, score = [], []
    for line in infile:
        words = line.strip().split('|||')
        tmp = []
        tmp.append([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[0].split()])
        tmp.append([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[1].split()])
        valid_data.append(tmp)
        score.append(float(words[2]))
    #print "Number of testing examples: ", len(valid_data)
    return valid_data,  score

def getTestingData(file_name, words_vocab):
    infile = open(file_name, 'r')
    testing_data, score = [], []
    for line in infile:
        words = line.strip().split('|||')
        tmp = []
        tmp.append([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[0].split()])
        tmp.append([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[1].split()])
        testing_data.append(tmp)
        score.append(float(words[2]))
    #print "Number of testing examples: ", len(testing_data)
    return testing_data,  score

def run_mlp(train_data, valid_data, valid_score, test_data, test_score, We_init, options):

    W_init = np.asarray((np.diag(np.ones(options.dim))), dtype='float32')

    g1batchindices = T.imatrix(); g2batchindices = T.imatrix()
    We_mask = T.fmatrix()

    # Create an instance of the MLP class
    mlp = Layer(We_init, W_init, T.tanh,  options.lamda_w, options.lamda_ww)

    #compute phrase vectors
    bigram_output = theano.function([g1batchindices, We_mask], mlp.output(g1batchindices, We_mask))

    cost = squared_error(mlp, g1batchindices, g2batchindices,  We_mask)

    cost = cost + mlp.word_reg

    updates = adagrad(cost, mlp.params, learning_rate=0.005, epsilon=1e-6)

    train_model = theano.function([g1batchindices, g2batchindices, We_mask], cost, updates=updates)

    # compute number of minibatches for training
    batch_size = int(options.batchsize)
    n_train_batches = int(len(train_data) * 1.0 // batch_size)

    iteration = 0

    max_iteration = options.epochs

    # --------------valid data format
    max_valid = 0
    for two in valid_data:
            for i in two:
                if len(i) > max_valid:
                    max_valid = len(i)
    valid_data_x = np.zeros((len(valid_data), max_valid), dtype='int32')
    valid_data_y = np.zeros((len(valid_data), max_valid), dtype='int32')

    for i, idata in enumerate(valid_data):
        for j, jdata in enumerate(idata[0]):
            valid_data_x[i, j] = jdata
        for j, jdata in enumerate(idata[1]):
            valid_data_y[i, j] = jdata

    # --------------test data format
    max_test = 0
    for two in test_data:
            for i in two:
                if len(i) > max_test:
                    max_test = len(i)
    test_data_x = np.zeros((len(test_data), max_test), dtype='int32')
    test_data_y = np.zeros((len(test_data), max_test), dtype='int32')

    for i, idata in enumerate(test_data):
        for j, jdata in enumerate(idata[0]):
            test_data_x[i, j] = jdata
        for j, jdata in enumerate(idata[1]):
            test_data_y[i, j] = jdata
    #    -------------------

    while iteration < max_iteration:
        iteration += 1

        seed = range(len(train_data))
        random.shuffle(seed)
        train_data = [train_data[i] for i in seed]

        score = valid_model(bigram_output, valid_data_x, valid_data_y, valid_score, We_mask_init)

        accuary = test_model(bigram_output, test_data_x, test_data_y, test_score, We_mask_init)

        print "iteration: {0}   valid_score: {1}   test_score: {2}".format(iteration, score[0], accuary[0])

        for minibatch_index in range(n_train_batches):

            train_data_batch = train_data[minibatch_index * batch_size : (minibatch_index + 1) * batch_size]

            max_train_batch = 0
            for two in train_data_batch:
                for i in two:
                    if len(i) > max_train_batch:
                        max_train_batch = len(i)

            train_data_x = np.zeros((len(train_data_batch), max_train_batch), dtype='int32')
            train_data_y = np.zeros((len(train_data_batch), max_train_batch), dtype='int32')

            for i, idata in enumerate([train_data_batch[i][0] for i in range(len(train_data_batch))]):
                for j, jdata in enumerate(idata):
                    train_data_x[i, j] = jdata

            for i, idata in enumerate([train_data_batch[i][1] for i in range(len(train_data_batch))]):
                for j, jdata in enumerate(idata):
                    train_data_y[i, j] = jdata

            train_model(train_data_x, train_data_y, We_mask_init)


class options(object):
    def __init__(self):
        self.epochs = 5
        self.lamda_ww = 0.0001
        self.lamda_w = 0.1
        self.batchsize = 100
    #    self.dim = 25


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    options = options()

    # parser.add_argument("", help)
    # parser.add_argument("-outfile", help="Output file name.")
    parser.add_argument("-batchsize", help="Size of batch.", type=int)
    parser.add_argument("-wordfile", help="Word embedding file.")
    # parser.add_argument("-epochs", help="Number of epochs in training.", type=int)
    parser.add_argument("-lamda_ww", help="Lambda for word embeddings.")
    parser.add_argument("-lamda_w", help="Lambda for parameters.")
    parser.add_argument("-dim", help="dimension of embeddings.")
    # parser.add_argument("-learner", help="update method.")

    args = parser.parse_args()
    #
    # options.outfile = args.outfile
    options.batchsize = float(args.batchsize)
    # options.epochs = float(args.epochs)
    options.lamda_ww = float(args.lamda_ww)
    options.lamda_w = float(args.lamda_w)
    options.dim = int(args.dim)
    # options.learner = args.learner

    print args.wordfile
    print args.batchsize, args.lamda_ww, args.lamda_w
    We_init, words_vocab = getVectors(args.wordfile, options)
    #We_init, words_vocab = getVectors('../embedding/paragram_vectors.txt', options)
    We_mask_init = np.ones((len(We_init), len(We_init[0])), dtype='float32')
    We_mask_init[0] = 0
  
    # train_data = getTrainingData(args.train, words_vocab)
    train_data = getTrainingData('../data/chinese/para_mt_train.txt', words_vocab)  # train_data is number

    valid_data, valid_score = getValidData('../data/chinese/para_mt_dev.txt', words_vocab)
    test_data, test_score = getTestingData('../data/chinese/para_mt_test.txt', words_vocab)

    run_mlp(train_data, valid_data, valid_score, test_data, test_score, We_init, options)
