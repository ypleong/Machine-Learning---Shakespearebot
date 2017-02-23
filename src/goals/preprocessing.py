from nltk.tokenize import word_tokenize
import pickle
import operator
import string

def process_data(filename):
    all_lines = []
    all_words = {}
    all_poems = []
    poem_temp = []
    with open(filename) as f:
        for line in f.readlines():
            line_tokens = [word.lower() for word in word_tokenize(line)]

            if len(line_tokens) > 1:
                for word in line_tokens:
                    if word in all_words:
                        all_words[word] += 1
                    else:
                        all_words[word] = 1
                all_lines.append(line_tokens)
                poem_temp += line_tokens
            elif len(line_tokens) == 1:
                all_poems.append(poem_temp)
                poem_temp = []

    all_poems.append(poem_temp)
    all_poems = all_poems[1:]

    return all_words, all_poems, all_lines


def compute_bigram_count(all_lines, all_words):
    bigram_list = {}
    for line in all_lines:
        for ind in range(len(line)-1):
            if (line[ind], line[ind+1]) in bigram_list:
                bigram_list[(line[ind], line[ind+1])] += 1
            else:
                bigram_list[(line[ind], line[ind + 1])] = 1

    bigram_list = sorted(bigram_list.items(), key=operator.itemgetter(1), reverse=True)

    for ind, (bigram, value) in enumerate(bigram_list):
        bigram_list[ind] = (bigram, all_words[bigram[0]], all_words[bigram[1]], value)

    return bigram_list


def replace_bigram(all_bigrams, all_lines, threshold=20):
    bigrams_that_matters = [item[0] for item in all_bigrams if item[-1] >= threshold]

    all_words = set()
    for line in all_lines:
        # new_line = []
        for ind in range(len(line) - 1):
            try:
                if (line[ind], line[ind + 1]) in bigrams_that_matters:
                    line[ind] = line[ind] + ' ' + line[ind + 1]
                    del line[ind+1]
            except IndexError:
                pass

        all_words.update(line)

    return all_lines, list(all_words)


def compute_rhythm_dictionary(all_lines):
    all_rhythm = set()
    poem = 0
    begin = 0
    stanza = 0
    tot_line = 0
    rhythm1 = []
    rhythm2 = []

    punctuations = list(string.punctuation)

    for line in all_lines:

        line = [item for item in line if item not in punctuations]

        if poem == 98:
            if stanza == 0:
                rhythm1.append(line[-1])
                tot_line += 1

                if tot_line == 5:
                    all_rhythm.update([tuple(sorted([rhythm1[0], rhythm1[2]])),
                                      tuple(sorted([rhythm1[1], rhythm1[3]])),
                                      tuple(sorted([rhythm1[2], rhythm1[4]]))])
                    rhythm1 = []
                    stanza += 1
                    tot_line = 0

            elif stanza < 3:
                if begin == 0:
                    rhythm1.append(line[-1])
                    begin = 1
                elif begin == 1:
                    rhythm2.append(line[-1])
                    begin = 0
                tot_line += 1

                if tot_line == 4:
                    all_rhythm.update([tuple(sorted(rhythm1)), tuple(sorted(rhythm2))])
                    tot_line = 0
                    stanza += 1
                    rhythm1 = []
                    rhythm2 = []

            else:
                rhythm1.append(line[-1])
                tot_line += 1

                if tot_line == 2:
                    tot_line = 0
                    all_rhythm.add(tuple(sorted(rhythm1)))
                    stanza = 0
                    rhythm1 = []
                    poem += 1

        elif poem == 125:
            rhythm1.append(line[-1])
            tot_line += 1

            if tot_line == 2:
                tot_line = 0
                all_rhythm.add(tuple(sorted(rhythm1)))
                stanza += 1
                rhythm1 = []

            if stanza == 6:
                poem += 1
                stanza = 0

        else:
            if stanza < 3:
                if begin == 0:
                    rhythm1.append(line[-1])
                    begin = 1
                elif begin == 1:
                    rhythm2.append(line[-1])
                    begin = 0
                tot_line += 1

                if tot_line == 4:
                    all_rhythm.update([tuple(sorted(rhythm1)), tuple(sorted(rhythm2))])
                    tot_line = 0
                    stanza += 1
                    rhythm1 = []
                    rhythm2 = []

            else:
                rhythm1.append(line[-1])
                tot_line += 1

                if tot_line == 2:
                    tot_line = 0
                    all_rhythm.add(tuple(sorted(rhythm1)))
                    stanza = 0
                    rhythm1 = []
                    poem += 1

    return all_rhythm


def convert_to_integer(all_words, all_lines):
    dictionary = {i: word for i, word in enumerate(all_words)}
    new_lines = []
    for line in all_lines:
        new_line = []
        for word in line:
            new_line.append(all_words.index(word))
        new_lines.append(new_line)

    return new_lines, dictionary


def convert_to_word(int_lines, dictionary):
    new_lines = []
    for line in int_lines:
        new_lines.append([dictionary[ind] for ind in line])

    return new_lines


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



all_words, all_poems, all_lines = process_data('../../project2data/shakespeare.txt')

all_bigrams = compute_bigram_count(all_lines, all_words)

all_rhythm = compute_rhythm_dictionary(all_lines)

training_line, training_symbols = replace_bigram(all_bigrams, all_lines, threshold=20)

training_line_int, dictionary = convert_to_integer(training_symbols, training_line)

dictionary_2 = dict((v,k) for k,v in dictionary.iteritems())

save_obj(training_line_int.reverse(), 'training_data')
save_obj(dictionary, 'word_dictionary')
save_obj(dictionary_2, 'word_dictionary_reverse')
save_obj(training_symbols, 'symbols')
save_obj(all_rhythm, 'rhythm')

print(dictionary_2)
print(all_rhythm)
