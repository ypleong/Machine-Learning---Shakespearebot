import numpy as np
import random
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
O = load_obj('observation_matrix')
word_dict = load_obj('word_dictionary')
A = load_obj('transition_matrix')
L = len(O)
D = len(O[0])
#Normalize columns of O to get P(y|x) from P(x|y) using Bayes Rule
O_norm = normalize_column_matrix(O,D,L)

#first_word = 'increase'
first_word = word_dict[0][random.randint(0,D-1)]
rand_sequence = [first_word]
num_word = word_dict[0].index(first_word)
P_word = O_norm[:,num_word]

state = []
print(sum(O[1]))
#Initialize first state based on first word
P_S = P_word
r_S = random.uniform(0,1)
for k in range(L):
    if r_S < sum(P_S[0:k+1]):
        state.append(k)
        break

M=8
prev_k = -1
#Update each state and emission
i = 0
syllable_count = 0
while syllable_count < 10:
    #Update each state based on previous state
    P_A = A[state[i]]
    print(P_A)
    r_A = random.uniform(0,1)
    for k in range(L):
        if r_A < sum(P_A[0:k+1]):
            state.append(k)
            break
    #Determine current emission based on current state
    P_O = O[state[i]]
    r_O = random.uniform(0,1)
    for k in range(D):
        if r_O < sum(P_O[0:k+1]):
            if (k == prev_k):
                state.pop()
                break
            else:
                new_word = word_dict[0][k]
                syllable_count += syllables_in_word(new_word)
                if (syllable_count > 10):
                    state.pop()
                    syllable_count -= syllables_in_word(new_word)
                else:
                    rand_sequence.append(new_word)
                    prev_k = k
                    i = i + 1
                    break

print(state)
print(rand_sequence)

                                                                                                                                
