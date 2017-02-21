from nltk.tokenize import word_tokenize
from nltk.tag.hmm import HiddenMarkovModelTagger, HiddenMarkovModelTrainer, _create_hmm_tagger
from nltk.probability import (DictionaryConditionalProbDist,
                              RandomProbDist)
import numpy as np
import random
import networkx as nx
import time
import matplotlib.pyplot as plt
import pickle
from nltk.corpus import cmudict
import string

phoneme_dict = dict(cmudict.entries())

def syllables_in_word(word):
    '''Attempts to count the number of syllables in the string argument 'word'.
    Limitation: word must be in the CMU dictionary (but that was a premise of the Exercise)
    "Algorithm": no. syllables == no. (0,1,2) digits in the dictionary entry, right??        
    '''
    # although listcomps may be readable, you can't insert print statements to instrument them!!
    if phoneme_dict.has_key(word):
        #return sum([ phoneme.count(str(num)) for phoneme in phoneme_dict[word] for num in range(3) ])
        return len( [ph for ph in phoneme_dict[word] if ph.strip(string.letters)] )   # more destructive; less efficient? NO! see timeit results in my comments below
    else:
        return 0


def syllables_in_text(text):
    '''Attempts to count the number of syllables in the string argument 'text'.
    Limitation: any "internal punctuation" must be part of the word. (it wouldn't get "this,and" correctly)
    Lets syllables_in_word do the heavy lifting.
    '''
    # ok, so apparently str.split(delim) only works for A SINGLE CHAR delim...
    # anything fancier, and you might want a regex (and its associated performance penalty)
    return sum([syllables_in_word(word.strip(string.punctuation)) for word in text.split()]) # - alternatives at http://stackoverflow.com/questions/265960/    

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


def visualize_transition_matrix_graph(A):
    G = nx.DiGraph()
    Asize = A.shape
    G.add_nodes_from(range(Asize[0]))

    edge_labels = {}

    for i in range(Asize[0]):
        for j in range(Asize[1]):
            G.add_edge(i, j, weight=A[i, j])
            edge_labels[(i, j)] = '{:.4f}'.format(A[i, j])

    pos = nx.spectral_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, width=2)

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_family='sans-serif')

    plt.axis('off')
    plt.savefig("weighted_graph.png")  # save as png
    plt.show()  # display


def plot_observation_bar(O):
    Osize = O.shape

    f, ax = plt.subplots(Osize[0], sharex=True, sharey=True)

    for ind in range(len(ax)):
        # print ind
        ax[ind].scatter(range(Osize[1]), O[ind], marker='.')
    # plt.setp(markerline, 'linewidth', 1)

    ax[0].set_title('Sharing both axes')
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.xlim(0, Osize[1])
    # plt.ylim(0, None)

    plt.savefig("observation_graph.png")  # save as png
    plt.show()  # display


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


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

all_words, all_poems, all_lines = process_data('../project2data/shakespeare.txt')

training = []
#for line in all_lines:
#    training.append([(i, None) for i in line])
#print(training)

for line in all_lines:
    empty_list=[]
    curr_syl = 0
    for i in line:
        curr_syl += syllables_in_word(i)
        empty_list.extend([(i+str(curr_syl), None)])
    training.append(empty_list)

states = range(10)
symbols = list(all_words)

priors = RandomProbDist(states)
A = DictionaryConditionalProbDist(
                    dict((state, RandomProbDist(states))
                                              for state in states))
O = DictionaryConditionalProbDist(
                    dict((state, RandomProbDist(symbols))
                                              for state in states))
model = HiddenMarkovModelTagger(symbols, states, A, O, priors)

# model = _create_hmm_tagger(states, symbols, A, O, pi)

#training = []
#for line in all_poems:
#    training.append([(i, None) for i in line])

trainer = HiddenMarkovModelTrainer(states, symbols)
curr_time = time.time()
hmm = trainer.train_unsupervised(training, model=model,max_iterations=8)
        
#seq = model.random_sample(random.Random(),7)
