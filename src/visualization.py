import numpy as np
import pickle
import matplotlib.pyplot as plt
import operator


def plot_matrix(A, O, word_dict, normalize=True):
    """
    Plot A and O
    :param A: state transition matrix
    :param O: state observation matrix
    :param word_dict: a dictionary of integer to words
    :param normalize: normalize probability for each state to between 0 and 1
    :return: None
    """
    Osize = O.shape
    Onew = O

    if normalize:
        Onew = np.zeros(Osize)
        Anew = np.zeros(A.shape)
        for row in range(Osize[0]):
            Onew[row, :] = O[row, :]/max(O[row, :])
            Anew[row, :] = A[row, :]/max(A[row, :])

    plt.imshow(Onew, aspect='auto', cmap='magma', interpolation='nearest')
    plt.colorbar(orientation='horizontal', aspect=100)
    plt.clim(vmin=0, vmax=1)
    plt.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(Onew[:, :100], aspect='auto', cmap='magma', interpolation='nearest', vmin=0.0, vmax=1.0)
    ax1.set_xticks(range(100))
    ax1.set_xticklabels(word_dict[:100], rotation=90)
    plt.show()  # display

    plt.matshow(A, aspect='auto', cmap='magma')
    plt.colorbar()
    plt.show()


def find_top_words_for_states(n, O, symbols):
    """
    Top n words for each state.
    Return dictionary with the state as the key
    and a list of tuple (word, probability) as the value

    :param n: top n words
    :param O: observation matrix
    :param symbols: a dictionary of integer to word
    :return a list of top 10 words with probability,
    and a list of top words that form 50% of probability
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


def load_obj(name):
    """
    Load data
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    """
    Save object
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def count_words(all_lines):
    """
    Count the frequency of each words in all_lines
    """
    words = {}
    for line in all_lines:
        for word in line:
            if word in words:
                words[word] += 1
            else:
                words[word] = 1

    new_words = sorted(words.items(), key=operator.itemgetter(1), reverse=True)

    return [item[0] for item in new_words]

#########################################################################
#                               Main code                               #
#########################################################################

# load all data
A_list = load_obj('./data/transition_matrix_line_20')
O_list = load_obj('./data/observation_matrix_line_20')
symbols = load_obj('./sonnet_preprocessing_data/symbols')
word_dict = load_obj('./sonnet_preprocessing_data/word_dictionary')
lines = load_obj('./sonnet_preprocessing_data/training_data')

A = np.array(A_list, dtype=np.float64)
O = np.array(O_list, dtype=np.float64)

# count the words
ordered_words_int = count_words(lines)

# reorder column according to wordcount
O = O[:, ordered_words_int]
ordered_words = [word_dict[ind] for ind in ordered_words_int]

# plot result
plot_matrix(A, O, ordered_words)

# count top 10 words for each hidden state
top_10_words, top_words_prob = find_top_words_for_states(10, O, symbols)

# print length of top 50% probability
for l in top_words_prob:
    print len(top_words_prob[l])

# print top 10 words in latex format
for state in top_10_words:
    print str(state) + ' & ' + ' & '.join([item[0] for item in top_10_words[state]]) + ' \\\\'
    print ' & ' + ' & '.join(['{:.4f}'.format(float(item[1])) for item in top_10_words[state]]) + ' \\\\ \\hline'


