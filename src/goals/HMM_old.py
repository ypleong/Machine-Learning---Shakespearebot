########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Richard Cheng, Andrew Kang, Avishek Dutta
# Description:  Set 5 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (i.e. run `python 2G.py`) to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.
            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            D:          Number of observations.
            A:          The transition matrix.
            O:          The observation matrix.
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''
        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for i in range(self.L)]

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    Output sequence corresponding to x with the highest
                        probability.
        '''
        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for i in range(self.L)] for j in range(M + 1)]
        seqs = [['' for i in range(self.L)] for j in range(M + 1)]

        #Initialize probabilities and sequences
        probs[0] = self.A_start
        for i in range(self.L):
            probs[0][i] = self.A_start[i]*self.O[i][x[0]]
            seqs[0][i] = str(i)

        #Implement Viterbi algorithm to populate sequences & associated probabilities
        for k in range(1,M):
            for j in range(self.L):
                dummy_probs = []
                for i in range(self.L):
                    dummy_probs.append(probs[k-1][i]*self.A[i][j]*self.O[j][x[k]])
                index = dummy_probs.index(max(dummy_probs))
                probs[k][j] = max(dummy_probs)
                seqs[k][j] = seqs[k-1][index] + str(j)
                
        

        #Return the maximum likelihood sequence
        max_seq_index = probs[M-1].index(max(probs[M-1]))
        max_seq = seqs[M-1][max_seq_index]

        return max_seq

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.
                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''
        M = len(x)      # Length of sequence.
        alphas = [[0. for i in range(self.L)] for j in range(M + 1)]

        #Initialize alphas
        for i in range(self.L):
            alphas[0][i] = self.A_start[i]
            alphas[1][i] = self.A_start[i]*self.O[i][x[0]]

        #Normalize first row of alphas
        if normalize:
            normalize_sum = sum(alphas[1])
            for i in range(self.L):
                alphas[1][i] = alphas[1][i]/normalize_sum
        #Update alphas based on A and O
        for j in range(1,M):
            for i in range(self.L):
                dummy_var = 0.0
                for k in range(self.L):
                    dummy_var = dummy_var + alphas[j][k]*self.A[k][i]
                alphas[j+1][i] = dummy_var*self.O[i][x[j]]
            #Normalize each row of alphas
            if normalize:
                normalize_sum = sum(alphas[j+1])
                for i in range(self.L):
                    alphas[j+1][i] = alphas[j+1][i]/normalize_sum
        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.
                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.
        '''
        M = len(x)      # Length of sequence.
        betas = [[0. for i in range(self.L)] for j in range(M + 1)]
        #Initialize betas
        for i in range(self.L):
            betas[M][i] = 1.0

        #Normalize Mth row of betas
        if normalize:
            normalize_sum = sum(betas[M])
            for i in range(self.L):
                betas[M][i] = betas[M][i]/normalize_sum
        #Update betas based on A and O
        for j in range(M-1,-1,-1):
            for i in range(self.L):
                dummy_var = 0.0
                for k in range(self.L):
                    dummy_var = dummy_var + betas[j+1][k]*self.A[i][k]*self.O[k][x[j]]
                betas[j][i] = dummy_var
            #Normalize each row of betas
            if normalize:
                normalize_sum = sum(betas[j])
                for i in range(self.L):
                        betas[j][i] = betas[j][i]/normalize_sum

        return betas

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        '''
        # Calculate each element of A using the M-step formulas.
        counts_A = [[0. for i in range(self.L)] for j in range(self.L)]
        states_A = [0. for i in range(self.L)]
        A = [[0. for i in range(self.L)] for j in range(self.L)]

        #Count transitions between states for each of the sequences in Y
        for i in range(len(Y)):
            for j in range(1,len(Y[i])):
                prev = Y[i][j-1]
                curr = Y[i][j]
                counts_A[prev][curr] += 1
                states_A[prev] += 1
        for i in range(self.L):
            for j in range(self.L):
                A[i][j] = counts_A[i][j]/states_A[i]

        self.A = A

        # Calculate each element of O using the M-step formulas.
        counts_O = [[0. for i in range(self.D)] for j in range(self.L)]
        states_O = [0. for i in range(self.L)]
        O = [[0. for i in range(self.D)] for j in range(self.L)]

        #Count observation-state pairs for each X-Y sequence pair
        for i in range(len(X)):
            for j in range(len(X[i])):
                counts_O[Y[i][j]][X[i][j]] += 1
                states_O[Y[i][j]] += 1 

        for i in range(self.L):
            for j in range(self.D):
                O[i][j] = counts_O[i][j]/states_O[i]
        
        self.O = O
        
        pass

    def unsupervised_learning(self, X):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
v        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.
        '''
        A =[[0. for i in range(self.L)] for l in range(self.L)]
        O =[[0. for i in range(self.D)] for l in range(self.L)]
        A_num =[[0. for i in range(self.L)] for l in range(self.L)]
        A_den = [0. for i in range(self.L)]
        O_num =[[0. for i in range(self.D)] for l in range(self.L)]
        O_den =[0. for i in range(self.L)]

        #Iterate EM algorithm 1000 times
        for N in range(100):
            for k in range(len(X)):
                #Compute first set of marginal probabilities (P1) for each sequence in X
                P1 = [[0. for i in range(self.L)] for j in range(len(X[k]))]
                alpha = self.forward(X[k],normalize=True)
                beta = self.backward(X[k],normalize=True)
                for i in range(len(X[k])):
                    prob_sum = 0.0
                    for j in range(self.L):
                        prob_sum = prob_sum +  alpha[i+1][j]*beta[i+1][j]
                    for j in range(self.L):
                        P1[i][j] = alpha[i+1][j]*beta[i+1][j]/prob_sum

                #Calculate necessary parameters to estimate O based on marginal probabilities
                for i in range(len(X[k])):
                    for j in range(self.L):
                        O_den[j] += P1[i][j]
                        O_num[j][X[k][i]] += P1[i][j]

                #Calculate second set of marginal probabilities (P2) for each sequence in X
                P2 = [[[0. for i in range(self.L)] for l in range(self.L)] for j in range(len(X[k]))]
                for i in range(0,len(X[k])):
                    prob_sum = 0.0
                    for prev in range(self.L):
                        for curr in range(self.L):
                            P2[i][prev][curr] = alpha[i][prev]*self.A[prev][curr]*self.O[curr][X[k][i]]*beta[i+1][curr]
                            prob_sum += alpha[i][prev]*self.A[prev][curr]*self.O[curr][X[k][i]]*beta[i+1][curr]
                    for prev in range(self.L):
                        for curr in range(self.L):
                            P2[i][prev][curr] = P2[i][prev][curr]/prob_sum

                #Calculate necessary parameters to estimate A based on marginal probabilities
                for i in range(1,len(X[k])):
                    for prev in range(self.L):
                        for curr in range(self.L):
                            A_den[prev] += P2[i][prev][curr]
                            A_num[prev][curr] += P2[i][prev][curr]
                        
            #Update A and O based on calculated probabilities
            for prev in range(self.L):
                for curr in range(self.L):
                    A[prev][curr] = A_num[prev][curr]/A_den[prev]
            for i in range(self.L):
                for j in range(self.D):
                    O[i][j] = O_num[i][j]/O_den[i]

            #Update A and O in the HMM object for this EM iteration                                   
            self.A = A
            self.O = O

        pass

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a string.
        '''
        emission = ''

        #Initialize state & emission length
        M = 20
        state = [0. for i in range(M+1)]
        state[0] = random.randint(0,self.L-1)
        P_O = self.O[state[0]]
        r_O = random.uniform(0,1)
        for k in range(self.D):
            if r_O < sum(P_O[0:k+1]):
                emission += str(k)
                break

        #Update each state and emission
        for i in range(1,M):
            #Update each state based on previous state
            P_A = self.A[state[i-1]]
            r_A = random.uniform(0,1)            
            for k in range(self.L):
                if r_A < sum(P_A[0:k+1]):
                    state[i] = k
                    break
            
            #Determine current emission based on current state
            P_O = self.O[state[i]]
            r_O = random.uniform(0,1)
            for k in range(self.D):
                if r_O < sum(P_O[0:k+1]):
                    emission += str(k)
                    break

        return emission

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''
        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the output sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''
        betas = self.backward(x)

        # beta_j(0) gives the probability of the output sequence. Summing
        # this over all states and then normalizing gives the total
        # probability of x paired with any output sequence, i.e. the
        # probability of x.
        #prob = sum(betas[0]) / self.L
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                                        for j in range(self.L)])
        return prob

def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learing.

    Arguments:
        X:          A list of variable length emission sequences 
        Y:          A corresponding list of variable length state sequences
                    Note that the elements in X line up with those in Y
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.
        n_states:   Number of hidden states to use in training.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X)

    return HMM

