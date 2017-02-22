from nltk.tokenize import word_tokenize
from nltk.tag.hmm import HiddenMarkovModelTagger, HiddenMarkovModelTrainer
from nltk.probability import (DictionaryConditionalProbDist,
                              RandomProbDist)
import numpy as np
import random
import time
import pickle
import operator
import string


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
    """return a LxD matrix"""
    A = [[random.random() for _ in range(D)] for _ in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    return np.array(A, dtype=np.float64)


def transition_matrix(hmm):
    trans_iter = (hmm._transitions[sj].prob(si)
                  for sj in hmm._states
                  for si in hmm._states)

    transitions_prob = np.fromiter(trans_iter, dtype=np.float64)
    N = len(hmm._states)
    return transitions_prob.reshape((N, N))


def observation_matrix(hmm):
    trans_iter = (hmm._outputs[sj].prob(si)
                  for sj in hmm._states
                  for si in hmm._symbols)

    transitions_prob = np.fromiter(trans_iter, dtype=np.float64)
    N = len(hmm._states)
    M = len(hmm._symbols)
    return transitions_prob.reshape((N, M))


def find_top_words_for_states(n, O, symbols):
    """
    Top n words for each state.
    Return dictionary with the state as the key
    and a list of tuple (word, probability) as the value
    """
    Osize = O.shape

    top_words = {}
    for state in range(Osize[0]):
        top_index = sorted(range(len(O[state])), key=lambda i: O[state, i], reverse=True)[:n]
        top_words[state] = [(symbols[ind], O[state, ind]) for ind in top_index]

    return top_words


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def compute_bigram_count(all_lines):
    bigram_list = {}
    for line in all_lines:
        for ind in range(len(line)-1):
            if (line[ind], line[ind+1]) in bigram_list:
                bigram_list[(line[ind], line[ind+1])] += 1
            else:
                bigram_list[(line[ind], line[ind + 1])] = 1

    return sorted(bigram_list.items(), key=operator.itemgetter(1), reverse=True)


def compute_rhythm_dictionary(all_lines):
    all_rhythm = set()
    poem = 0
    begin = 0
    stanza = 0
    tot_line = 0
    rhythm1 = []
    rhythm2 = []

    punctuations = list(string.punctuation)

    for line in all_lines:

        line = [item for item in line if item not in punctuations]

        if poem == 98:
            if stanza == 0:
                rhythm1.append(line[-1])
                tot_line += 1

                if tot_line == 5:
                    all_rhythm.update([tuple(sorted([rhythm1[0], rhythm1[2]])),
                                      tuple(sorted([rhythm1[1], rhythm1[3]])),
                                      tuple(sorted([rhythm1[2], rhythm1[4]]))])
                    rhythm1 = []
                    stanza += 1
                    tot_line = 0

            elif stanza < 3:
                if begin == 0:
                    rhythm1.append(line[-1])
                    begin = 1
                elif begin == 1:
                    rhythm2.append(line[-1])
                    begin = 0
                tot_line += 1

                if tot_line == 4:
                    all_rhythm.update([tuple(sorted(rhythm1)), tuple(sorted(rhythm2))])
                    tot_line = 0
                    stanza += 1
                    rhythm1 = []
                    rhythm2 = []

            else:
                rhythm1.append(line[-1])
                tot_line += 1

                if tot_line == 2:
                    tot_line = 0
                    all_rhythm.add(tuple(sorted(rhythm1)))
                    stanza = 0
                    rhythm1 = []
                    poem += 1

        elif poem == 125:
            rhythm1.append(line[-1])
            tot_line += 1

            if tot_line == 2:
                tot_line = 0
                all_rhythm.add(tuple(sorted(rhythm1)))
                stanza += 1
                rhythm1 = []

            if stanza == 6:
                poem += 1
                stanza = 0

        else:
            if stanza < 3:
                if begin == 0:
                    rhythm1.append(line[-1])
                    begin = 1
                elif begin == 1:
                    rhythm2.append(line[-1])
                    begin = 0
                tot_line += 1

                if tot_line == 4:
                    all_rhythm.update([tuple(sorted(rhythm1)), tuple(sorted(rhythm2))])
                    tot_line = 0
                    stanza += 1
                    rhythm1 = []
                    rhythm2 = []

            else:
                rhythm1.append(line[-1])
                tot_line += 1

                if tot_line == 2:
                    tot_line = 0
                    all_rhythm.add(tuple(sorted(rhythm1)))
                    stanza = 0
                    rhythm1 = []
                    poem += 1

    return all_rhythm


all_words, all_poems, all_lines = process_data('../project2data/shakespeare.txt')

all_bigrams = compute_bigram_count(all_lines)

all_rhythm = compute_rhythm_dictionary(all_lines)

states = range(10)
symbols = list(all_words)

L = len(states)
D = len(symbols)

# Randomly initialize and normalize matrix A.
# A = create_random_matrix(L, L)
# visualize_transition_matrix_graph(A)

# Randomly initialize and normalize matrix O.
# O = create_random_matrix(L, D)
# find_top_words_for_states(10, O, symbols)

# pi = [1. / L for _ in range(L)]

priors = RandomProbDist(states)
A = DictionaryConditionalProbDist(
                dict((state, RandomProbDist(states))
                     for state in states))
O = DictionaryConditionalProbDist(
                dict((state, RandomProbDist(symbols))
                     for state in states))
model = HiddenMarkovModelTagger(symbols, states,
                                A, O, priors)

# model = _create_hmm_tagger(states, symbols, A, O, pi)

training = []
for line in all_poems:
    training.append([(i, None) for i in line])

trainer = HiddenMarkovModelTrainer(states, symbols)
curr_time = time.time()
hmm = trainer.train_unsupervised(training, model=model,
                                 max_iterations=100)
print time.time() - curr_time

top_10_words = find_top_words_for_states(10, observation_matrix(hmm), symbols)
save_obj(top_10_words, 'top_10_words')
save_obj(observation_matrix(hmm), 'observation_matrix')
save_obj(transition_matrix(hmm), 'transition_matrix')
# plot_observation_bar(observation_matrix(hmm))

