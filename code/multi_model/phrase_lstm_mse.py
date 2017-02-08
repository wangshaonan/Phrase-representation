from copy import copy

import numpy as np
import theano
import theano.tensor as  T
from collections import OrderedDict
import copy
import argparse
import math
import random
from scipy.stats import spearmanr
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)

srng = RandomStreams(SEED)


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


def lstm_layer(tparams, state_below, options, mask=None): # tparams, emb_x, options, x_mask
    nsteps = state_below.shape[0] #N: sentence's length
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]  # B: batch size
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, tparams['lstm_U'])  # h*U: (B*E)*(E*4E)=B*4E
        preact += x_  # B*4E + B*4E

        i = T.nnet.sigmoid(_slice(preact, 0, options.dim))
        f = T.nnet.sigmoid(_slice(preact, 1, options.dim))
        o = T.nnet.sigmoid(_slice(preact, 2, options.dim))
        c = T.tanh(_slice(preact, 3, options.dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (T.dot(state_below, tparams['lstm_W']) +
                   tparams['lstm_b'])  # x*W+b: N*B*E * E*4E=N*B*4E

    dim_proj = options.dim
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[T.alloc(numpy_floatX(0.),  #B*E
                                                      n_samples,
                                                      dim_proj),
                                              T.alloc(numpy_floatX(0.),
                                                      n_samples,
                                                      dim_proj)],
                                name='layers',
                                n_steps=nsteps)
    return rval[0]  # h : Step*N*E


def dropout_layer(state_before, use_noise, srng):
    proj = T.switch(use_noise,
                    (state_before *
                     srng.binomial(state_before.shape,
                                   p=0.5, n=1,
                                   dtype=state_before.dtype)),
                    state_before * 0.5)
    return proj


def init_params(options, We_init):

    params = OrderedDict()
    # embedding
    params['Wemb'] = copy.copy(We_init)

    # Init the LSTM parameter:
    W = np.concatenate([ortho_weight(options.dim),
                        ortho_weight(options.dim),
                        ortho_weight(options.dim),
                        ortho_weight(options.dim)], axis=1)
    params['lstm_W'] = W
    U = np.concatenate([ortho_weight(options.dim),
                        ortho_weight(options.dim),
                        ortho_weight(options.dim),
                        ortho_weight(options.dim)], axis=1)
    params['lstm_U'] = U

    b = np.zeros((4 * options.dim))
    params['lstm_b'] = b.astype('float32')

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def squared_error(x, y):  #x: B*E
    return T.sum((x - y) ** 2)


def cosine_dis(x, y):
    g1g2 = (x * y).sum(axis=1)
    g1g2norm = T.sqrt(T.sum(x ** 2, axis=1)) * T.sqrt(T.sum(y ** 2, axis=1))
    return 1 - T.mean(g1g2 / g1g2norm)


numpy_floatX = lambda x: np.asarray(x, dtype='float32')


def sgd_updates_adadelta(norm,params,cost,rho=0.95,epsilon=1e-9,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value(),dtype='float32')
        exp_sqr_grads[param] = theano.shared(value=(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)

    # this is place you should think of gradient clip using the l2-norm
    g2 = 0.
    clip_c = 1.
    for g in gparams:
        g2 += (g**2).sum()
    # is_finite = T.or_(T.isnan(g2), T.isinf(g2))
    new_grads = []
    for g in gparams:
        new_grad = T.switch(g2>(clip_c**2),g/T.sqrt(g2)*clip_c,g)
        new_grads.append(new_grad)
    gparams = new_grads

    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = T.cast(rho * exp_sg + (1 - rho) * T.sqr(gp),'float32')
        updates[exp_sg] = up_exp_sg
        step = T.cast(-(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp, 'float32')
        updates[exp_su] = T.cast(rho * exp_su + (1 - rho) * T.sqr(step), 'float32')
        stepped_param = param + step
        if norm == 1:
            if (param.get_value(borrow=True).ndim == 2) and param.name!='Words':
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
                scale = T.cast(desired_norms / (1e-7 + col_norms),'float32')
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param
        elif norm == 0:
            updates[param] = stepped_param
        else:
            updates[param] = stepped_param
    return updates


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


def test_model(output_x, output_y, test_data_x, test_data_y, test_mask_x, test_mask_y, test_score):

    predication1 = output_x(test_data_x, test_mask_x)
    predication2 = output_y(test_data_y, test_mask_y)

    p1p2 = (predication1 * predication2).sum(axis=1)  # B
    p1p2norm = np.sqrt((predication1 ** 2).sum(axis=1)) * np.sqrt((predication2 ** 2).sum(axis=1))
    cos = p1p2 / p1p2norm  # cosine

    corr = spearmanr(cos, test_score)
    return corr


def valid_model(output_x, output_y, valid_data_x, valid_data_y, valid_mask_x, valid_mask_y, valid_score):

    predication1 = output_x(valid_data_x, valid_mask_x)  # B*E
    predication2 = output_y(valid_data_y, valid_mask_y)

    p1p2 = (predication1 * predication2).sum(axis=1)  # B
    p1p2norm = np.sqrt((predication1 ** 2).sum(axis=1)) * np.sqrt((predication2 ** 2).sum(axis=1))
    cos = p1p2 / p1p2norm  # cosine

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
        tmp_vec /= math.sqrt((tmp_vec ** 2).sum() + 1e-6)
        # vectors.append(list(tmp_vec))
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
    # init parameters of lstm : Wemb, lstm_U, lstm_W, lstm_b
    params = init_params(options, We_init)
    # shared value
    tparams = init_tparams(params)

    use_noise = theano.shared(numpy_floatX(0.))

    x = T.lmatrix('x')  # N*B*E
    x_mask = T.matrix('x_mask', dtype='float32')

    y = T.lmatrix('y')
    y_mask = T.matrix('y_mask', dtype='float32')

    nx_timesteps = x.shape[0] #N
    nx_samples = x.shape[1] #B

    ny_timesteps = y.shape[0]
    ny_samples = y.shape[1]

    emb_x = tparams['Wemb'][x.flatten()].reshape([nx_timesteps,
                                                  nx_samples,
                                                  options.dim])
    emb_y = tparams['Wemb'][y.flatten()].reshape([ny_timesteps,
                                                  ny_samples,
                                                  options.dim])

    proj_x = lstm_layer(tparams, emb_x, options, x_mask)  #N*B*E
    proj_y = lstm_layer(tparams, emb_y, options, y_mask)

    # proj_x = dropout_layer(proj_x, use_noise, srng)
    # proj_y = dropout_layer(proj_y, use_noise, srng)

    output_x = theano.function([x, x_mask], proj_x[-1])
    output_y = theano.function([y, y_mask], proj_y[-1])

    #cost = squared_error(proj_x[-1], proj_y[-1])
    cost = cosine_dis(proj_x[-1], proj_y[-1])

    train_model = theano.function([x, x_mask, y, y_mask], cost, updates=sgd_updates_adadelta(0, list(tparams.values()), cost))

    # train_model = theano.function([x, x_mask, y, y_mask], cost, updates=adagrad(cost, list(tparams.values())))

    # compute number of minibatches for training
    batch_size = int(options.batchsize)
    n_train_batches = int(len(train_data) * 1.0 // batch_size)

    iteration = 0
    lrate = numpy_floatX(0.0001)
    max_iteration = options.epochs

    # --------------valid data format
    max_valid = 0
    for two in valid_data:
        for i in two:
            if len(i) > max_valid:
                max_valid = len(i)
    valid_data_x = np.zeros((max_valid, len(valid_data)), dtype='int64')
    valid_data_y = np.zeros((max_valid, len(valid_data)), dtype='int64')
    valid_mask_x = np.zeros((max_valid, len(valid_data)), dtype='float32')
    valid_mask_y = np.zeros((max_valid, len(valid_data)), dtype='float32')

    for i, idata in enumerate(valid_data):
        for j, jdata in enumerate(idata[0]):
            valid_data_x[j, i] = jdata
            valid_mask_x[j, i] = 1
        for j, jdata in enumerate(idata[1]):
            valid_data_y[j, i] = jdata
            valid_mask_y[j, i] = 1

    # -------------------

    # --------------test data format
    max_test = 0
    for two in test_data:
        for i in two:
            if len(i) > max_test:
                max_test = len(i)
    test_data_x = np.zeros((max_test, len(test_data)), dtype='int64')
    test_data_y = np.zeros((max_test, len(test_data)), dtype='int64')
    test_mask_x = np.zeros((max_test, len(test_data)), dtype='float32')
    test_mask_y = np.zeros((max_test, len(test_data)), dtype='float32')

    for i, idata in enumerate(test_data):
        for j, jdata in enumerate(idata[0]):
            test_data_x[j, i] = jdata
            test_mask_x[j, i] = 1
        for j, jdata in enumerate(idata[1]):
            test_data_y[j, i] = jdata
            test_mask_y[j, i] = 1
    # -------------------

    use_noise.set_value(1.)
    while iteration < max_iteration:
        cost = 0
        iteration += 1

        random.shuffle(train_data)

        # temp1 = copy.copy(tparams['Wemb'].get_value())
        # temp2 = copy.copy(tparams['lstm_W'].get_value())
        for minibatch_index in range(n_train_batches):
            # assert (tparams['Wemb'].get_value() == temp1).all()
            # assert (tparams['lstm_W'].get_value() == temp2).all()

            train_data_batch = train_data[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

            max_train_batch = 0
            for two in train_data_batch:
                for i in two:
                    if len(i) > max_train_batch:
                        max_train_batch = len(i)

            train_data_x = np.zeros((max_train_batch, len(train_data_batch)), dtype='int64')
            train_data_y = np.zeros((max_train_batch, len(train_data_batch)), dtype='int64')
            train_mask_x = np.zeros((max_train_batch, len(train_data_batch)), dtype='float32')
            train_mask_y = np.zeros((max_train_batch, len(train_data_batch)), dtype='float32')

            for i, idata in enumerate([train_data_batch[i][0] for i in range(len(train_data_batch))]):
                for j, jdata in enumerate(idata):
                    train_data_x[j, i] = jdata
                    train_mask_x[j, i] = 1

            for i, idata in enumerate([train_data_batch[i][1] for i in range(len(train_data_batch))]):
                for j, jdata in enumerate(idata):
                    train_data_y[j, i] = jdata
                    train_mask_y[j, i] = 1

            cost += train_model(train_data_x, train_mask_x, train_data_y, train_mask_y)

        print 'cost: %f' % cost

        score = valid_model(output_x, output_y, valid_data_x, valid_data_y, valid_mask_x, valid_mask_y, valid_score)

        accuary = test_model(output_x, output_y, test_data_x, test_data_y, test_mask_x, test_mask_y, test_score)

        print "iteration: {0}   valid_score: {1}   test_score: {2}".format(iteration, score[0], accuary[0])


class options(object):
    def __init__(self):
        self.epochs = 5
        self.lamda_ww = 0.0001
        self.lamda_w = 0.1
        self.batchsize = 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    options = options()

    # parser.add_argument("", help)
    # parser.add_argument("-outfile", help="Output file name.")
    parser.add_argument("-batchsize", help="Size of batch.", type=int)
    parser.add_argument("-wordfile", help="Word embedding file.")
    # parser.add_argument("-train", help="Training data file.")
    # parser.add_argument("-test", help="Testing data file.")
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
    #options.lamda_ww = float(args.lamda_ww)
    #options.lamda_w = float(args.lamda_w)
    # options.learner = args.learner
    print args.wordfile
    print args.batchsize
    We_init, words_vocab = getVectors(args.wordfile)
    options.nvocab = len(words_vocab)
    options.dim = int(args.dim)

    train_data = getTrainingData('../data/english/ppdb_train.txt', words_vocab)  # train_data is number

    valid_data, valid_score = getValidData('../data/english/ppdb_dev.txt', words_vocab)
    test_data,  test_score = getTestingData('../data/english/ppdb_test.txt', words_vocab)
    run_mlp(train_data, valid_data, valid_score, test_data, test_score, We_init, options)
