import random
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.tag.hmm import HiddenMarkovModelTagger, HiddenMarkovModelTrainer, _create_hmm_tagger
from nltk.probability import (DictionaryConditionalProbDist,RandomProbDist)
#from nltk.tag.hmm import create_hmm_tagger 

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
                                                                                                            

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


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

top_10_words = load_obj('top_10_words')
O = load_obj('observation_matrix')
A = load_obj('transition_matrix')

print top_10_words

all_words, all_poems, all_lines = process_data('../project2data/shakespeare.txt')
states = range(10)
symbols = list(all_words)

L = len(states)
D = len(symbols)

#priors = RandomProbDist(states)
pi = [1. / L for _ in range(L)]
#model = HiddenMarkovModelTagger(symbols, states, A, O, priors)
model = _create_hmm_tagger(states, symbols, A, O, pi)

seq = model.random_sample(random.Random(),7)

print(seq)
