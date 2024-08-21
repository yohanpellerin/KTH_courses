import os
import pickle
from parse_dataset import Dataset
from dep_parser import Parser
from logreg import LogisticRegression
import numpy as np


class TreeConstructor:
    """
    This class builds dependency trees and evaluates using unlabeled arc score (UAS) and sentence-level accuracy
    """
    def __init__(self, parser):
        self.__parser = parser

    def build(self, model, words, tags, ds):
        """
        Build the dependency tree using the logistic regression model `model` for the sentence containing
        `words` pos-tagged by `tags`
        
        :param      model:  The logistic regression model
        :param      words:  The words of the sentence
        :param      tags:   The POS-tags for the words of the sentence
        :param      ds:     Training dataset instance having the feature maps
        """
        #
        # YOUR CODE HERE
        #
        small_ds = Dataset()
        small_ds.copy_feature_maps(ds)
        i, stack, pred_tree = 0, [], [0]*(len(words)) # Input configuration
        m = self.__parser.SH
        while m != None :
            i,stack,pred_tree = self.__parser.move(i,stack,pred_tree,m)
            small_ds.add_datapoint(words, tags, i, stack, m)
            x,_ = small_ds.to_arrays()
            x = np.concatenate((np.ones((len(x), 1)), x), axis=1)
            valid_moves = self.__parser.valid_moves(i, stack, pred_tree)
            if len(valid_moves) == 0:
                m=None
            elif len(valid_moves) == 1:
                m = valid_moves[0]
            else:
                best_prob, best_class = -float('inf'), None
                second_best_prob, second_best_class = -float('inf'), None
                for c in range(model.CLASSES):
                    prob = model.conditional_log_prob(c, x[-1])
                    if prob > best_prob:
                        second_best_prob = best_prob
                        second_best_class = best_class
                        best_prob = prob
                        best_class = c
                    elif prob > second_best_prob:
                        second_best_prob = prob
                        second_best_class = c
                if best_class in valid_moves:
                    m = best_class
                else:
                    m = second_best_class
        return pred_tree

            
        

        
   

        

    def evaluate(self, model, test_file, ds):
        """
        Evaluate the model on the test file `test_file` using the feature representation given by the dataset `ds`
        
        :param      model:      The model to be evaluated
        :param      test_file:  The CONLL-U test file
        :param      ds:         Training dataset instance having the feature maps
        """
        #
        # YOUR CODE HERE
        #
        sentence_accuracy = 0
        correct_arcs = 0
        total_arcs = 0
        nb_sentences = 0
        with open(test_file, encoding='utf-8') as f:
            for w,tags,tree,_ in self.__parser.trees(f):
                buil_tree = self.build(model, w, tags, ds)
                nb_sentences += 1
                correct = True
                for i in range(len(tree)):
                    total_arcs += 1
                    if tree[i] != buil_tree[i]:
                        correct = False
                    else:
                        correct_arcs += 1
                          
                if correct:
                    sentence_accuracy += 1
        print('Sentence-level accuracy: {:.2f}%'.format(sentence_accuracy / nb_sentences * 100))
        print('UAS: {:.2f}%'.format(correct_arcs / total_arcs * 100))




if __name__ == '__main__':

    # Create parser
    p = Parser()

    # Create training dataset
    ds = p.create_dataset("en-ud-train-projective.conllu", train=True)

    # Train LR model
    if os.path.exists('model.pkl'):
        # if model exists, load from file
        print("Loading existing model...")
        lr = pickle.load(open('model.pkl', 'rb'))
        ds.to_arrays()
    else:
        # train model using minibatch GD
        lr = LogisticRegression()
        lr.fit(*ds.to_arrays())
        pickle.dump(lr, open('model.pkl', 'wb'))
    
    # Create test dataset
    test_ds = p.create_dataset("en-ud-dev-projective.conllu")
    # Copy feature maps to ensure that test datapoints are encoded in the same way
    test_ds.copy_feature_maps(ds)
    # Compute move-level accuracy
    lr.classify_datapoints(*test_ds.to_arrays())

    # Compute UAS and sentence-level accuracy
    t = TreeConstructor(p)
    t.evaluate(lr, 'en-ud-dev-projective.conllu', ds)
