from nltk.tokenize import word_tokenize
from nltk.tag.hmm import HiddenMarkovModelTagger, HiddenMarkovModelTrainer
from nltk.probability import (DictionaryConditionalProbDist,
                              RandomProbDist)
import numpy as np
import random
import time
import pickle


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

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


all_words, all_poems, all_lines = process_data('../../project2data/shakespeare.txt')

symbols = list(all_words)
D = len(symbols)

dictionary = [[0. for _ in range(D)] for _ in range(2)]
for i in range(D):
    dictionary[1][i] = i
    dictionary[0][i] = symbols[i]


training = []
for line in all_lines:
    empty_list = []
    for i in line:
        index = dictionary[0].index(i)
        empty_list.append(index)
    empty_list.reverse()
    training.append(empty_list)

save_obj(training, 'training_data')
save_obj(dictionary, 'word_dictionary')
save_obj(symbols, 'symbols')
