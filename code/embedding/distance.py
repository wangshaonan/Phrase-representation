#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import struct
import numpy as np


N = 10  # number of closest words that will be shown


def load_data(f):
    # file type f
    vocab = []
    feature = []
    for line in f:
        line = line.strip().split()
        if len(line) < 5:
            continue
	vocab.append(line[0])
        feature.append([float(i) for i in line[1:]])		
    feature = np.array(feature)

    return vocab, feature


def calc_distance(target, vocab, feature):
    try:
        i = vocab.index(target)
        rank = (feature * feature[i]).sum(axis=1)
    except ValueError:
        # target does not exist
        rank = None

    return rank


def load_freq(f):
    freqlist = []
    for line in f:
        word, freq = line.split()
        freqlist.append(freq)

    return freqlist


if __name__ == '__main__':

    filename = sys.argv[1]

    try:
        with open(filename, 'rb') as f:
            vocab, feature = load_data(f)
    except IOError:
        print("Input file not found\n")
        sys.exit(-1)

    while True:
        target = raw_input("Enter word: ")
        rank = calc_distance(target, vocab, feature)
        if rank is None:
            print("Out of dictionary word!")
            continue

        indexed_rank = []
        for i, r in enumerate(rank):
            indexed_rank.append((r, i))

        print("word\tdistance")
        for r in sorted(indexed_rank, key=lambda x: x[0], reverse=True)[1:N]:
            distance, i = r
            if len(sys.argv) == 3:
                print("{}\t{:06f}\t{}".format(vocab[i], distance, freqlist[i]))
            else:
                print("{}\t{:06f}".format(vocab[i], distance))

        print("")


