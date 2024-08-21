from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.1  # The learning rate.
    CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
    MAX_ITERATIONS = 1000 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1000 # Minibatch size (only for minibatch gradient descent)

    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, use_weight=False,theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        self.__use_weight = use_weight
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)



    # ----------------------------------------------------------------------


    def sigmoid(self, z):
        """
        The logistic function.
        """
        return expit(z)


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """
        
        if label==1:
                prod = self.sigmoid(np.dot(self.x[datapoint],self.theta))
        else:
                prod = 1-self.sigmoid(np.dot(self.x[datapoint],self.theta))
        return prod


    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """

        # YOUR CODE HERE
        return np.dot(self.x.T,(self.sigmoid(np.dot(self.x,self.theta))-self.y))/self.DATAPOINTS
    
    def count_label_positive(self):
        if self.__use_weight:
            self.weight_negative = np.sum(self.y)/self.DATAPOINTS
            self.weight_positive = 1-self.weight_negative
        else:
            self.weight_negative = 1
            self.weight_positive = 1
    
    def compute_loss_val(self):
        epsilon = 1e-10
        h_theta = self.sigmoid(np.dot(self.x_val, self.theta))
        loss = -np.sum(np.log(h_theta + epsilon) * self.y_val*self.weight_positive + np.log(1 - h_theta + epsilon) * (1 - self.y_val)*self.weight_negative) / len(self.x_val)
        return loss
    
    def compute_loss(self):
        epsilon = 1e-10  # Small value to prevent taking the logarithm of zero
        h_theta = self.sigmoid(np.dot(self.x, self.theta))
        loss = -np.sum(np.log(h_theta + epsilon) * self.y *self.weight_positive+ np.log(1 - h_theta + epsilon) * (1 - self.y)*self.weight_negative) / self.DATAPOINTS
        return loss
    
    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """
        
        # YOUR CODE HERE
    
        return self.x[minibatch].T.dot(self.sigmoid(np.dot(self.x[minibatch],self.theta))-self.y[minibatch])/len(minibatch)


    def compute_gradient(self, i):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """

        # YOUR CODE HERE
        return self.x[i].T.dot(self.weight_negative*self.sigmoid(np.dot(self.x[i],self.theta))-self.y[i]+(self.weight_positive-self.weight_negative)*self.y[i]*(1-self.sigmoid(np.dot(self.x[i],self.theta))))
    
    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        # YOUR CODE HERE
        converge = False
        n=0
        while not converge and n<self.MAX_ITERATIONS:
            #select a random datapoint
            i = random.randint(0,self.DATAPOINTS-1)
            self.gradient = self.compute_gradient(i)
            self.theta = self.theta - self.LEARNING_RATE * self.gradient
            n+=1
            self.update_plot(np.sum(np.square(self.gradient)))
            if all(abs(grad) < self.CONVERGENCE_MARGIN for grad in self.gradient):
                converge = True

    def stochastic_fit_with_early_stopping(self):
        """
        Performs Stochastic Gradient Descent.
        """
        self.init_plot(self.FEATURES)
        self.x_val = self.x[:int(self.DATAPOINTS*0.1)]
        self.y_val = self.y[:int(self.DATAPOINTS*0.1)]
        self.x = self.x[int(self.DATAPOINTS*0.1):]
        self.y = self.y[int(self.DATAPOINTS*0.1):]
        self.loss_val = np.inf
        self.DATAPOINTS_train = len(self.x)
        
        # YOUR CODE HERE
        converge = False
        k=0
        n=0
        while not converge and n<self.MAX_ITERATIONS:
            #select a random datapoint
            i = random.randint(0,self.DATAPOINTS_train-1)
            self.gradient = self.compute_gradient(i)
            self.loss_val_previous = self.loss_val
            self.loss_val = np.sum(np.square(self.compute_gradient_for_all_val()))
            self.theta = self.theta - self.LEARNING_RATE * self.gradient
            n+=1
            self.update_plot(np.sum(np.square(self.gradient)))
            if all(abs(grad) < self.CONVERGENCE_MARGIN for grad in self.gradient):
                converge = True

            #early stopping
            
            if n>100 and self.loss_val_previous<self.loss_val:
                k+=1
            else:
                k=0
            if k==5:
                converge = True
                print('early stopping')
                

    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_plot(self.FEATURES)
        # YOUR CODE HERE
        n=0
        converge = False
        #while every element of the gradient is greater than the convergence margin
        while not converge and  n<self.MAX_ITERATIONS:
            #shuffle the data
            indices = np.random.permutation(self.DATAPOINTS)
            self.x = self.x[indices]
            self.y = self.y[indices]
            for i in range(0,self.DATAPOINTS,self.MINIBATCH_SIZE):
                if i+self.MINIBATCH_SIZE>self.DATAPOINTS:
                    mini_batch = [j for j in range (i,self.DATAPOINTS)]
                else:
                    mini_batch = [j for j in range (i,i+self.MINIBATCH_SIZE)]
                self.gradient = self.compute_gradient_minibatch(mini_batch)
                self.theta = self.theta - self.LEARNING_RATE * self.gradient
            n+=1
            self.update_plot(np.sum(np.square(self.gradient)))
            if all(abs(grad) < self.CONVERGENCE_MARGIN for grad in self.gradient):
                converge = True


    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)

        # YOUR CODE HERE
        n=0
        converge = False
        #while every element of the gradient is greater than the convergence margin
        while not converge and  n<self.MAX_ITERATIONS:
            self.gradient = self.compute_gradient_for_all()
            self.theta = self.theta - self.LEARNING_RATE * self.gradient
            n+=1
            self.update_plot(np.sum(np.square(self.gradient)))
            if all(abs(grad) < self.CONVERGENCE_MARGIN for grad in self.gradient):
                converge = True
        
    def fit_with_early_stopping(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)
        self.x_val = self.x[:int(self.DATAPOINTS*0.1)]
        self.y_val = self.y[:int(self.DATAPOINTS*0.1)]
        self.x = self.x[int(self.DATAPOINTS*0.1):]
        self.y = self.y[int(self.DATAPOINTS*0.1):]
        self.loss_val = np.inf
        self.DATAPOINTS_train = len(self.x)
        self.count_label_positive()
        # YOUR CODE HERE
        k=0
        n=0
        converge = False
        #while every element of the gradient is greater than the convergence margin
        while not converge and  n<self.MAX_ITERATIONS:
            self.gradient = self.compute_gradient_for_all()
            self.loss_val_previous = self.loss_val
            self.loss_val = self.compute_loss_val()
            
            self.theta = self.theta - self.LEARNING_RATE * self.gradient
            n+=1
            self.update_plot(self.compute_loss())
            if all(abs(grad) < self.CONVERGENCE_MARGIN for grad in self.gradient):
                converge = True
            #early stopping
            if n>5 and (self.loss_val_previous-self.loss_val)/self.loss_val_previous<0.0001:
                k+=1
            else:
                k=0
            if k==5:
                converge = True
                print('early stopping')
            if n%10==0:
                print('loss_val:',self.loss_val)


    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:');

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))
        self.index_false = []

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1
            if predicted != self.y[d]:
                self.index_false.append(d)

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):

        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(max(self.i),1) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)
        
        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
         [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],[ 0,0 ], [ 0,0 ], [ 1,1 ],
        [ 0,0 ], [ 0,0 ], [ 1,0 ],[ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [ 1, 0, 0, 0, 0, 0, 0,
          0, 0, 1, 1, 0, 0, 0, 1, 0]
    print('minibatch')
    b = BinaryLogisticRegression(x, y)
    b.minibatch_fit()
    b.print_result()
    b.classify_datapoints(x, y)
    print('stochoastic')

    c = BinaryLogisticRegression(x, y)
    c.stochastic_fit()
    c.print_result()
    c.classify_datapoints(x, y)
    print('batch')



    d = BinaryLogisticRegression(x, y)
    d.fit()
    d.print_result()
    d.classify_datapoints(x, y)
    


if __name__ == '__main__':
    main()
