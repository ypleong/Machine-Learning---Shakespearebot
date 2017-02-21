import pickle
import networkx as nx
import matplotlib.pyplot as plt


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