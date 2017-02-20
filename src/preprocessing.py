from nltk.tokenize import word_tokenize
from nltk.tag.hmm import demo_bw, _create_hmm_tagger, HiddenMarkovModelTrainer
import numpy as np
import random
import time

all_lines = []
all_words = set()
all_poems = []
poem_temp = []
with open('../project2data/shakespeare.txt') as f:
    for line in f.readlines():
        line_tokens = word_tokenize(line)

        if len(line_tokens) > 1:
            all_words.update(line_tokens)
            all_lines.append(line_tokens)
            poem_temp += line_tokens
        elif len(line_tokens) == 1:
            all_poems.append(poem_temp)
            poem_temp = []

    all_poems.append(poem_temp)
    all_poems = all_poems[1:]

states = range(100)
symbols = list(all_words)

L = len(states)
D = len(symbols)

# Randomly initialize and normalize matrix A.
# random.seed(100)
A = [[random.random() for i in range(L)] for j in range(L)]

for i in range(len(A)):
    norm = sum(A[i])
    for j in range(len(A[i])):
        A[i][j] /= norm

# Randomly initialize and normalize matrix O.
O = [[random.random() for i in range(D)] for j in range(L)]

for i in range(len(O)):
    norm = sum(O[i])
    for j in range(len(O[i])):
        O[i][j] /= norm

pi = [1. / L for _ in range(L)]

model = _create_hmm_tagger(states, symbols, A, O, pi)

training = []
for line in all_poems:
    training.append([(i, None) for i in line])

trainer = HiddenMarkovModelTrainer(states, symbols)
curr_time = time.time()
hmm = trainer.train_unsupervised(training, model=model,
                                 max_iterations=1000)
print time.time() - curr_time
print hmm
# demo_bw()