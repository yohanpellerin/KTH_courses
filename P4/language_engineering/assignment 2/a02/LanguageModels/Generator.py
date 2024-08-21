import math
import argparse
import codecs
from collections import defaultdict
import random

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):
    
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                # YOUR CODE HERE
                for i in range(self.unique_words):
                    _,word,unigram_count = f.readline().strip().split(' ')
                    self.index[word] = i
                    self.word[i] = word
                    self.unigram_count[self.index[self.word[i]]] = int(unigram_count)
                bigram = f.readline().strip().split(' ')
                while bigram[0] != "-1":
                    self.bigram_prob[int(bigram[0])][int(bigram[1])] = float(bigram[2])
                    bigram = f.readline().strip().split(' ')
                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and sampling from the distribution
        of the language model.
        """ 
        # YOUR CODE HERE
        list_words =[w]
        if self.index.get(w) is None:
                print("Unknown word")
        while n > 0:
            bigram = self.bigram_prob.get(self.index.get(w))
            if bigram is None:
                    # random choice
                    w = random.choice(list(self.index.keys()))
                    list_words.append(w)
                    n -= 1
            else:
                    #choose the next word according to the log probability of each word
                    exp_values = {nombre: math.exp(log_prob) for nombre, log_prob in bigram.items()}
                    somme_totale = sum(exp_values.values())
                    if somme_totale == 0:
                        w = random.choice(list(self.index.keys()))
                        list_words.append(w)
                        n -= 1
                    else:
                        nombre_aleatoire = random.uniform(0, somme_totale)
                        somme_cumulative = 0
                        for nombre, probabilite in exp_values.items():
                            somme_cumulative += probabilite
                            if somme_cumulative >= nombre_aleatoire:
                                list_words.append(self.word[nombre])
                                w = self.word[nombre]
                                n -= 1
                                break
        print(' '.join(list_words))                        


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=5)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    #print(generator.bigram_prob)
    #print(generator.index)
    #print(generator.word)
    generator.generate(arguments.start,arguments.number_of_words)

if __name__ == "__main__":
    main()
