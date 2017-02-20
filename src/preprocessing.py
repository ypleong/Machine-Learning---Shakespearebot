from nltk.tokenize import word_tokenize
from nltk.tag.hmm import demo_bw, _create_hmm_tagger, HiddenMarkovModelTrainer
import numpy as np
import random
import time
from hmmlearn.hmm import MultinomialHMM


def process_data(filename):
    all_lines = []
    all_words = set()
    all_poems = []
    poem_temp = []
    with open(filename) as f:
        for line in f.readlines():
            line_tokens = [word.lower() for word in word_tokenize(line)]

            if len(line_tokens) > 1:
                all_words.update(line_tokens)
                all_lines.append(line_tokens)
                poem_temp += line_tokens
            elif len(line_tokens) == 1:
                all_poems.append(poem_temp)
                poem_temp = []

    all_poems.append(poem_temp)
    all_poems = all_poems[1:]

    return all_words, all_poems, all_lines


def create_random_matrix(L, D):
    """return a LxD matrix as a list of list"""
    A = [[random.random() for _ in range(D)] for _ in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    return A

all_words, all_poems, all_lines = process_data('../project2data/shakespeare.txt')

states = range(100)
symbols = list(all_words)

L = len(states)
D = len(symbols)

# Randomly initialize and normalize matrix A.
A = create_random_matrix(L, L)

# Randomly initialize and normalize matrix O.
O = create_random_matrix(L, D)

pi = [1. / L for _ in range(L)]

model = _create_hmm_tagger(states, symbols, A, O, pi)

training = []
for line in all_poems:
    training.append([(i, None) for i in line])

trainer = HiddenMarkovModelTrainerSubClass(states, symbols)
curr_time = time.time()
hmm = trainer.train_unsupervised(training, model=model,
                                 max_iterations=1000)
print time.time() - curr_time
print hmm
# demo_bw()