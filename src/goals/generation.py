import numpy as np
import random
import pickle
from nltk.corpus import cmudict
import string
from countsyl import count_syllables

phoneme_dict = dict(cmudict.entries())


def syllables_in_word(word):
    #Attempts to count the number of syllables in the string argument 'word'.
    if phoneme_dict.has_key(word):
        return len( [ph for ph in phoneme_dict[word] if ph.strip(string.letters)] )  
    else:
        return 0


def normalize_column_matrix(M, D, L):
    """return a LxD matrix"""
    for i in range(D):
        column_sum = 0.0
        for j in range(L):
            column_sum += M[j][i]
        for j in range(L):
            M[j][i] = M[j][i]/column_sum
            
    return np.array(M, dtype=np.float64)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#Load in observation matrix and word dictionary (int,word pairs)
O = load_obj('./data/observation_matrix_line_20')
word_dict_rev = load_obj('word_dictionary_reverse')
word_dict = load_obj('word_dictionary')
A = load_obj('./data/transition_matrix_line_20')
rhythm_dict = load_obj('rhythm')
HMM = load_obj('./data/HMM_line_25')

L = len(O)
D = len(O[0])

rhyme_pair = [0. for _ in range(7)]
rand_selector = [0. for _ in range(7)]

for i in range(7):
    rhyme_pair[i] = random.sample(rhythm_dict, 1)[0]
    rand_selector[i] = random.randint(0, 1)

for l in range(14):
    # Set first word
    if l == 0 or l == 1:
        first_word = rhyme_pair[l][rand_selector[l]]
        first_index = word_dict_rev[first_word]
    elif l == 2 or l == 3:
        if rand_selector[l-2] == 1:
            first_word = rhyme_pair[l-2][0]
            first_index = word_dict_rev[first_word]
        else:
            first_word = rhyme_pair[l-2][1]
            first_index = word_dict_rev[first_word]
    elif l == 4 or l == 5:
        first_word = rhyme_pair[l-2][rand_selector[l-2]]
        first_index = word_dict_rev[first_word]
    elif l == 6 or l == 7:
        if rand_selector[l-4] == 1:
            first_word = rhyme_pair[l-4][0]
            first_index = word_dict_rev[first_word]
        else:
            first_word = rhyme_pair[l-4][1]
            first_index = word_dict_rev[first_word]
    elif l == 8 or l == 9:
        first_word = rhyme_pair[l-4][rand_selector[l-4]]
        first_index = word_dict_rev[first_word]
    elif l == 10 or l == 11:
        if rand_selector[l-6] == 1:
            first_word = rhyme_pair[l-6][0]
            first_index = word_dict_rev[first_word]
        else:
            first_word = rhyme_pair[l-6][1]
            first_index = word_dict_rev[first_word]
    elif l == 12:
        first_word = rhyme_pair[l-6][rand_selector[l-6]]
        first_index = word_dict_rev[first_word]
    elif l == 13:
        if rand_selector[l-7] == 1:
            first_word = rhyme_pair[l-7][0]
            first_index = word_dict_rev[first_word]
        else:
            first_word = rhyme_pair[l-7][1]
            first_index = word_dict_rev[first_word]

    # Get P(y|x) given first x
    P_yx = [0. for _ in range(L)]
    alphas = HMM.forward([first_index], normalize=True)
    betas = HMM.backward([first_index], normalize=True)
    for curr in range(L):
        P_yx[curr] = alphas[1][curr]*betas[0][curr]
    
    norm = sum(P_yx)
    for curr in range(len(P_yx)):
        P_yx[curr] /= norm

    state = []

    # Initialize first state based on first word
    P_S = P_yx
    r_S = random.uniform(0,1)
    for k in range(L):
        if r_S < sum(P_S[0:k+1]):
            state.append(k)
            break

    prev_k = -1
            
    # Update each state and emission
    i = 0
    syllable_count = count_syllables(first_word)
    if syllable_count == 0:
        syllable_count = syllables_in_word(first_word)
    rand_sequence=[first_word]
    while syllable_count < 10:
        # Update each state based on previous state
        P_A = A[state[i]]
        r_A = random.uniform(0,1)
        for k in range(L):
            if r_A < sum(P_A[0:k+1]):
                state.append(k)
                break
        # Determine current emission based on current state
        P_O = O[state[i]]
        r_O = random.uniform(0,1)
        for k in range(D):
            if r_O < sum(P_O[0:k+1]):
                if k == prev_k:
                    state.pop()
                    break
                else:
                    new_word = word_dict[k]
                
                    curr_syllable = count_syllables(new_word)
                    if curr_syllable == 0:
                        curr_syllable = syllables_in_word(new_word)
                    syllable_count += curr_syllable
                
                    if syllable_count > 10:
                        state.pop()
                        syllable_count -= count_syllables(new_word)
                        break
                    else:
                        rand_sequence.append(new_word)
                        prev_k = k
                        break
    rand_sequence.reverse()
    print(' '.join(rand_sequence))
