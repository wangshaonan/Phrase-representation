import numpy as np

vocab = set()
for line in open('../data/vocab/data.100k.vocab'):
    line = line.strip()
    vocab.add(line)

print len(vocab)
uk = np.zeros(100)
num = 0
outfile = open('vector.sg100.100k', 'w')
for line in open('vector.sg100'):
    words = line.strip().split()
    if len(words) < 10:
        continue
    if not words[0] in vocab:
	uk += np.array([float(i) for i in words[1:]])
	num += 1
        continue
    outfile.write(line)
uk = uk/num
outfile.write('UUUNKKK '+' '.join([str(i) for i in uk]))
