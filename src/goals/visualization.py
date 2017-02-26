import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import operator


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


def plot_matrix(A, O, word_dict, normalize=True):
    Osize = O.shape
    Onew = O
    Anew = A

    if normalize:
        Onew = np.zeros(Osize)
        Anew = np.zeros(A.shape)
        for row in range(Osize[0]):
            Onew[row, :] = O[row, :]/max(O[row, :])
            Anew[row, :] = A[row, :]/max(A[row, :])

    # plt.imshow(Onew, aspect='auto', cmap='magma', interpolation='nearest')
    # plt.colorbar(orientation='horizontal', aspect=100)
    # plt.clim(vmin=0, vmax=1)
    # plt.tight_layout()
    # plt.savefig("observation_graph.png")  # save as png
    # plt.show()


    fig, ax1 = plt.subplots(1, 1)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    ax1.imshow(Onew[:, :100], aspect='auto', cmap='magma', interpolation='nearest', vmin=0.0, vmax=1.0)
    # fig.colorbar(ax1, cax=cbar_ax)
    ax1.set_xticks(range(100))
    ax1.set_xticklabels(word_dict[:100], rotation=90)
    # ax1.tight_layout()
    plt.savefig("observation_graph.png")  # save as png
    plt.show()  # display

    # plt.imshow(Onew[:, 1000:2000], extent=[1000, 1999, 19, 0], aspect='auto', cmap='magma', interpolation='nearest')
    # plt.colorbar(orientation='horizontal', aspect=100)
    # plt.clim(vmin=0, vmax=1)
    # plt.tight_layout()
    # plt.savefig("observation_graph.png")  # save as png
    # plt.show()

    # plt.imshow(Onew[:, 2000:3000], extent=[2000, 2999, 19, 0], aspect='auto', cmap='magma',
    #            interpolation='nearest')
    # plt.colorbar(orientation='horizontal', aspect=100)
    # plt.clim(vmin=0, vmax=1)
    # plt.tight_layout()
    # plt.savefig("observation_graph.png")  # save as png
    # plt.show()
    #
    # plt.imshow(Onew[:, 3000:], extent=[3000, Osize[1], 19, 0], aspect='auto', cmap='magma',
    #            interpolation='nearest')
    # plt.colorbar(orientation='horizontal', aspect=20)
    # plt.clim(vmin=0, vmax=1)
    # plt.tight_layout()
    # plt.savefig("observation_graph.png")  # save as png
    # plt.show()

    plt.matshow(A, aspect='auto', cmap='magma')
    plt.colorbar()
    plt.savefig("transition_graph.png")  # save as png
    plt.show()


def find_top_words_for_states(n, O, symbols):
    """
    Top n words for each state.
    Return dictionary with the state as the key
    and a list of tuple (word, probability) as the value
    """
    Osize = O.shape

    top_words = {}
    top_words_prob = {}
    for state in range(Osize[0]):
        top_index = sorted(range(len(O[state])), key=lambda i: O[state, i], reverse=True)
        top_words[state] = [(symbols[ind], O[state, ind]) for ind in top_index[0:n]]

        curr_prob = O[state, top_index[0]]
        top_words_prob_temp = []
        ind = 0
        while curr_prob < 0.5:
            top_ind = top_index[ind]
            top_words_prob_temp.append((symbols[top_ind], O[state, top_ind]))
            ind += 1
            curr_prob += O[state, top_index[ind]]

        top_words_prob[state] = top_words_prob_temp

    return top_words, top_words_prob


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def count_words(all_lines):
    words = {}
    for line in all_lines:
        for word in line:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1

    new_words = sorted(words.items(), key=operator.itemgetter(1), reverse=True)

    return [item[0] for item in new_words]


A_list = load_obj('./data/transition_matrix_line_20')
O_list = load_obj('./data/observation_matrix_line_20')
symbols = load_obj('symbols')
word_dict = load_obj('word_dictionary')

lines = load_obj('training_data')
ordered_words_int = count_words(lines)

A = np.array(A_list, dtype=np.float64)
O = np.array(O_list, dtype=np.float64)

O = O[:, ordered_words_int]
ordered_words = [word_dict[ind] for ind in ordered_words_int]

# plot_matrix(A, O, ordered_words)

top_10_words, top_words_prob = find_top_words_for_states(10, O, symbols)

for list in top_words_prob:
    print len(top_words_prob[list])

for state in top_10_words:
    print str(state) + ' & ' + ' & '.join([item[0] for item in top_10_words[state]]) + ' \\\\'
    print ' & ' + ' & '.join(['{:.4f}'.format(float(item[1])) for item in top_10_words[state]]) + ' \\\\ \\hline'

# save_obj(top_10_words, 'top_10_words')
# print(top_10_words)


