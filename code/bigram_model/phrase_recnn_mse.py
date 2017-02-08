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
    def output(self, input_x, input_y):
        lin_output = (T.dot(T.concatenate((self.We[input_x], self.We[input_y]), axis=1), self.W))
        return self.activation(lin_output)

def squared_error(mlp, x1, x2, y1, y2):
    return T.sum((mlp.output(x1, x2) - mlp.output(y1, y2)) ** 2)

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

def test_model(mlp_output, test_data, test_score):

    test_data_x1 = [i[0] for i in test_data]
    test_data_x2 = [i[1] for i in test_data]

    test_data_y1 = [i[2] for i in test_data]
    test_data_y2 = [i[3] for i in test_data]

    predication1 = mlp_output(test_data_x1, test_data_x2)
    predication2 = mlp_output(test_data_y1, test_data_y2)

    # for i in range(len(predication1)):
    #     cos.append(1 - spatial.distance.cosine(predication1[i].tolist(), predication2[i].tolist())) #cosine simialrity for corresponding line

    p1p2 = (predication1*predication2).sum(axis=1) #B
    p1p2norm = np.sqrt((predication1 ** 2).sum(axis = 1)) * np.sqrt((predication2 ** 2).sum(axis=1))
    cos = p1p2/p1p2norm  #cosine

    corr = spearmanr(cos, test_score)
    return corr

def valid_model(mlp_output, valid_data, valid_score):

    valid_data_x1 = [i[0] for i in valid_data]
    valid_data_x2 = [i[1] for i in valid_data]

    valid_data_y1 = [i[2] for i in valid_data]
    valid_data_y2 = [i[3] for i in valid_data]

    predication1 = mlp_output(valid_data_x1, valid_data_x2)
    predication2 = mlp_output(valid_data_y1, valid_data_y2)

    p1p2 = (predication1*predication2).sum(axis=1) #B
    p1p2norm = np.sqrt((predication1 ** 2).sum(axis = 1)) * np.sqrt((predication2 ** 2).sum(axis=1))
    cos = p1p2/p1p2norm  #cosine

    corr = spearmanr(cos, valid_score)
    return corr

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

def getTrainingData(file_name, words_vocab):
    infile = open(file_name, 'r')
    training_data = []
    for line in infile:
        words = line.strip().split('|||')
        tmp = []
        tmp.append([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[0].split()])
        tmp.append([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[1].split()])
        training_data.append(tmp)

#    print "Number of training examples: ", len(training_data)
    return training_data

def getValidData(file_name, words_vocab):
    infile = open(file_name, 'r')
    valid_data, score = [], []
    for line in infile:
        words = line.strip().split('|||')
        tmp = []
        tmp.extend([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[0].split()])
        tmp.extend([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[1].split()])
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
        tmp.extend([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[0].split()])
        tmp.extend([words_vocab[i] if i in words_vocab else words_vocab['UUUNKKK']  for i in words[1].split()])
        testing_data.append(tmp)
        score.append(float(words[2]))
    #print "Number of testing examples: ", len(testing_data)
    return testing_data,  score

def run_mlp(train_data, valid_data, valid_score, test_data, test_score, We_init, options):

    tmp = np.diag(np.ones(options.dim, dtype='float32'))
    W_init = np.asarray(np.concatenate((tmp, tmp), axis=0))

    g1batchindices = T.lvector(); g2batchindices = T.lvector()
    p1batchindices = T.lvector(); p2batchindices = T.lvector()

    # Create an instance of the MLP class
    mlp = Layer(We_init, W_init, T.tanh,  options.lamda_w, options.lamda_ww)

    #compute phrase vectors
    bigram_output = theano.function([g1batchindices, g2batchindices], mlp.output(g1batchindices, g2batchindices))

    cost = squared_error(mlp, g1batchindices, g2batchindices, p1batchindices, p2batchindices)

    cost = cost + mlp.word_reg

    updates = adagrad(cost, mlp.params, learning_rate=0.005, epsilon=1e-6)

    train_model = theano.function([g1batchindices, g2batchindices, p1batchindices, p2batchindices], cost, updates=updates)

    # compute number of minibatches for training
    batch_size = int(options.batchsize)
    n_train_batches = int(len(train_data) * 1.0 // batch_size)

    iteration = 0

    max_iteration = options.epochs

    while iteration < max_iteration:
        iteration += 1

        seed = range(len(train_data))
        random.shuffle(seed)
        train_data = [train_data[i] for i in seed]

        score = valid_model(bigram_output, valid_data, valid_score)

        accuary = test_model(bigram_output, test_data, test_score)

        print "iteration: {0}   valid_score: {1}   test_score: {2}".format(iteration, score[0], accuary[0])

        for minibatch_index in range(n_train_batches):

            train_data_batch = train_data[minibatch_index * batch_size : (minibatch_index + 1) * batch_size]
            train_data_batch_x1 = [i[0][0] for i in train_data_batch]
            train_data_batch_x2 = [i[0][1] for i in train_data_batch]
            train_data_batch_y1 = [i[1][0] for i in train_data_batch]
            train_data_batch_y2 = [i[1][1] for i in train_data_batch]
            train_model(train_data_batch_x1, train_data_batch_x2, train_data_batch_y1, train_data_batch_y2)


class options(object):
    def __init__(self):
        self.epochs = 5
        self.lamda_ww = 0.0001
        self.lamda_w = 0.1
        self.batchsize = 100
        self.dim = 50


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #
    options = options()
    #
    # # parser.add_argument("", help)
    # # parser.add_argument("-outfile", help="Output file name.")
    parser.add_argument("-batchsize", help="Size of batch.", type=int)
    parser.add_argument("-wordfile", help="Word embedding file.")
    # # parser.add_argument("-epochs", help="Number of epochs in training.", type=int)
    parser.add_argument("-lamda_ww", help="Lambda for word embeddings.")
    parser.add_argument("-lamda_w", help="Lambda for parameters.")
    parser.add_argument("-dim", help="dimension of embeddings.")
    # # parser.add_argument("-learner", help="update method.")
    #
    args = parser.parse_args()
    # #
    # # options.outfile = args.outfile
    options.batchsize = float(args.batchsize)
    # # options.epochs = float(args.epochs)
    options.lamda_ww = float(args.lamda_ww)
    options.lamda_w = float(args.lamda_w)
    options.dim = int(args.dim)
    # # options.learner = args.learner
    #
    print args.wordfile
    print args.batchsize, args.lamda_ww, args.lamda_w
    We_init, words_vocab = getVectors(args.wordfile)
    options.dim = len(We_init[0])

    train_data = getTrainingData('../data/english/bigram_train.jn.txt', words_vocab)  # train_data is number

    valid_data, valid_score = getValidData('../data/english/bigram_dev.jn.txt', words_vocab)
    test_data, test_score = getTestingData('../data/english/bigram_test.jn.txt', words_vocab)    

    run_mlp(train_data, valid_data, valid_score, test_data, test_score, We_init, options)
