import os
import argparse
import time
import string
import numpy as np
from halo import Halo
from sklearn.neighbors import NearestNeighbors


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2018 by Dmytro Kalpakchi and Johan Boye.
"""


##
## @brief      Class for creating word vectors using Random Indexing technique.
## @author     Dmytro Kalpakchi <dmytroka@kth.se>
## @date       November 2018
##
class RandomIndexing(object):

    def __init__(self, filenames, dimension=100, non_zero=10, non_zero_values=list([-1, 1]), left_window_size=3, right_window_size=3):
        self.__sources = filenames
        self.__vocab = set()
        self.__dim = dimension
        self.__non_zero = non_zero
        # there is a list call in a non_zero_values just for Doxygen documentation purposes
        # otherwise, it gets documented as "[-1,"
        self.__non_zero_values = non_zero_values
        self.__lws = left_window_size
        self.__rws = right_window_size
        self.__cv = None
        self.__rv = None
        

    def clean_line(self, line):
        # YOUR CODE HERE
        for i in line:
            if not i.isalpha() and i != " ":
                line = line.replace(i, "")
        return line.split()

    def text_gen(self):
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)

    def build_vocabulary(self):
        # YOUR CODE HERE
        for line in self.text_gen():
            for word in line:
                #check if word is a word and not already in the vocabulary
                if word not in self.__vocab and word.isalpha():
                    self.__vocab.add(word)
        self.write_vocabulary()
       
    @property
    def vocabulary_size(self):
        return len(self.__vocab)


    def create_word_vectors(self):
        # YOUR CODE HERE

        #creation of the random vectors
        self.__rv = {}
        for word in self.__vocab:
            # Initialize an array of zeros of length self.__dim
            self.__rv[word] = np.zeros(self.__dim)

            # Randomly select indices to set to non-zero values
            non_zero_indices = np.random.choice(self.__dim, self.__non_zero, replace=False)

            # Set the selected indices to non-zero values
            self.__rv[word][non_zero_indices] = np.random.choice(self.__non_zero_values, self.__non_zero)

        #creation of the context vectors filled with zeros
        self.__cv = {}
        for word in self.__vocab:
            self.__cv[word] = np.zeros(self.__dim)
        
        #loop through the cleaned text and update the context vectors

        for line in self.text_gen():
            for i in range(len(line)):
                for j in range(i - self.__lws, i + self.__rws + 1):
                    if j >= 0 and j < len(line) and i != j:
                        self.__cv[line[i]] += self.__rv[line[j]]


    def find_nearest(self, words, k=5, metric='cosine'):
        # YOUR CODE HERE
        neighbors = []
        for word in words:
            if word not in self.__vocab:
                neighbors.append([None])
            else:
                nn = NearestNeighbors(n_neighbors=k, metric=metric)
                data = np.array([self.__cv[w] for w in self.__vocab])
                nn.fit(data)
                distance,indice = nn.kneighbors([self.__cv[word]],k, return_distance=True)
                for j in range (k):
                    neighbors.append([(self.__i2w[indice[0][j]],distance[0][j])])
        return neighbors

    def get_word_vector(self, word):
        # YOUR CODE HERE
        return self.__cv[word] if word in self.__cv else None

    def vocab_exists(self):
        return os.path.exists('vocab.txt')

    def read_vocabulary(self):
        vocab_exists = self.vocab_exists()
        if vocab_exists:
            with open('vocab.txt') as f:
                for line in f:
                    self.__vocab.add(line.strip())
        self.__i2w = list(self.__vocab)
        self.__w2i = {w: i for i, w in enumerate(self.__i2w)}
        return vocab_exists

    def write_vocabulary(self):
        print("Writing vocabulary to a file")
        with open('vocab.txt', 'w') as f:
            for w in self.__vocab:
                f.write('{}\n'.format(w))
        self.__i2w = list(self.__vocab)
        self.__w2i = {w: i for i, w in enumerate(self.__i2w)}

    def train(self):
        spinner = Halo(spinner='arrow3')

        if self.vocab_exists():
            spinner.start(text="Reading vocabulary...")
            start = time.time()
            self.read_vocabulary()
            spinner.succeed(text="Read vocabulary in {}s. Size: {} words".format(round(time.time() - start, 2), ri.vocabulary_size))
        else:
            spinner.start(text="Building vocabulary...")
            start = time.time()
            self.build_vocabulary()
            spinner.succeed(text="Built vocabulary in {}s. Size: {} words".format(round(time.time() - start, 2), ri.vocabulary_size))
        
        spinner.start(text="Creating vectors using random indexing...")
        start = time.time()
        self.create_word_vectors()
        spinner.succeed("Created random indexing vectors in {}s.".format(round(time.time() - start, 2)))

        spinner.succeed(text="Execution is finished! Please enter words of interest (separated by space):")


    ##
    ## @brief      Trains word embeddings and enters the interactive loop, where you can 
    ##             enter a word and get a list of k nearest neighours.
    ##
    def train_and_persist(self):
        self.train()
        self.write_to_file()
        self.interact()


    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("random_indexing.txt", 'w') as f:
                f.write("{} {}\n".format(self.vocabulary_size, self.__dim))
                for w in self.__i2w:
                    #print(w, self.__cv[w])
                    f.write(w + " " + " ".join(map(str, self.__cv[w])) + "\n")
        except:
            print("Error: failing to write the model to the file")
        
        

    @classmethod
    def load(cls, filename):
        """
        Load the model from a file
        """
        try:
            with open(filename, 'r') as f:
                V, H = map(int, f.readline().strip().split())
                ri = RandomIndexing([], H)
                ri.__w2i = {}
                ri.__cv = {}
                ri.__vocab = set()
                for i in range(V):
                    line = f.readline().strip().split()
                    w = line[0]
                    ri.__vocab.add(w)
                    ri.__w2i[w] = i
                    ri.__cv[w] = np.array(list(map(float, line[1:])))
                ri.__i2w = list(ri.__vocab)
                return ri
        except:
            print("Error: failing to load the model from the file")
            return None
        
    def interact(self):
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text)
            print(neighbors)
            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Indexing word embeddings')
    parser.add_argument('-fv', '--force-vocabulary', action='store_true', help='regenerate vocabulary')
    parser.add_argument('-c', '--cleaning', action='store_true', default=False)
    parser.add_argument('-s', '--save', default='random_indexing.txt', help='Filename where word vectors are saved')
    parser.add_argument('-co', '--cleaned_output', default='cleaned_example.txt', help='Output file name for the cleaned text')
    args = parser.parse_args()

    if os.path.exists(args.save):
        print("Loading the model from the file {}".format(args.save))
        ri = RandomIndexing.load(args.save)
        if ri:
            ri.interact()
    else:

        if args.force_vocabulary:
            os.remove('vocab.txt')

        if args.cleaning:
            ri = RandomIndexing(['example.txt'])
            with open(args.cleaned_output, 'w') as f:
                for part in ri.text_gen():
                    
                    f.write("{}\n".format(" ".join(part)))
        else:
            dir_name = "data"
            filenames = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name)]
            ri = RandomIndexing(filenames)
            ri.train_and_persist()
