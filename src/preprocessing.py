from nltk.tokenize import word_tokenize
from nltk.tag.hmm import demo_bw, _create_hmm_tagger
import numpy as np

all_lines = []
all_words = set()
with open('../project2data/shakespeare.txt') as f:
    for line in f.readlines():
        line_tokens = word_tokenize(line)
        if len(line_tokens) > 0:
            all_lines.append(line_tokens)

        if len(line_tokens) > 1:
            all_words.update(line_tokens)


print line

# states = ['bull', 'bear', 'static']
# symbols = ['up', 'down', 'unchanged']
# A = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]], np.float64)
# B = np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]], np.float64)
# pi = np.array([0.5, 0.2, 0.3], np.float64)
#
# model = _create_hmm_tagger(states, symbols, A, B, pi)

# demo_bw()