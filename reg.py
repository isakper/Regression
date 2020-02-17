import random
import numpy as np


# teorin kommer från https://medium.com/ml-ai-study-group/vectorized-implementation-of-cost-functions-and-gradient-vectors-linear-regression-and-logistic-31c17bca9181


class Lin_reg_layer:

    def __init__(self, num_in, num_out, bias_node = True):
        '''
        Creates a linear regression layer.
        :param num_in: How many input nodes. Does not include biased node
        :param num_out: How many output nodes
        :param bias_node: True if you want to add a biased node. False if not
        '''
        self._bias = bias_node
        if bias_node == True: # Creates a coefficient matrix with coefficients for bias nodes.
            self._coef = self.new_coef(num_in+1, num_out)
        else:
            self._coef = self.new_coef(num_in, num_out)


    def new_coef(self, dim_one, dim_two):
        '''
        Creates a matrix with random float values.

        :param dim_one: How many rows
        :param dim_two: how many columns
        :return: The matrix
        '''
        matrix = [[] for _ in range(dim_one)]
        for n in range(dim_one):
            for m in range(dim_two):
                matrix[n].append(random.random())
        return matrix

    def _add_bias(self,x):
        '''
        Adds a column of ones to matrix x
        :param x: a matrix
        :return: x with bias nodes
        '''
        x_b = [[] for _ in range(len(x))]
        for n in range(len(x)):
            x_b[n] = [1] + x[n]
        return x_b

    def feed_for(self, x):
        '''
        Does the feed forward process. Input some data and this function returns predicted value(s) using the coefficients.
        :param x: input matrix
        :return: predicted value(s)
        '''
        if self._bias == True:
            x_b = self._add_bias(x)
        else:
            x_b = x
        return np.matmul(x_b, self._coef)

    def grad_descent(self, x, y, lamb):
        '''
        Does a gradient descent step, changing the coefficients
        :param x: input data. Data must be in a nested list
        :param y: output data. Data must be in a nested list
        :param lamb: how big the step is
        '''

        #teorin kommer från https://medium.com/ml-ai-study-group/vectorized-implementation-of-cost-functions-and-gradient-vectors-linear-regression-and-logistic-31c17bca9181
        if self._bias == True:
            x_b = self._add_bias(x)
        else:
            x_b = x
        self._coef -= lamb/(len(x)) * np.matmul(np.transpose(x_b), np.matmul(x_b, self._coef) - y)

    def coef_ret(self):
        '''return coefficients'''
        return self._coef

    def cost(self, x, y):
        '''
        Calculates and returns cost
        :param x: input data. Data must be in a nested list
        :param y: output data. Data must be in a nested list
        :return:  the cost
        '''
        if self._bias == True:
            x_b = self._add_bias(x)
        else:
            x_b = x

        cost = 1/(2*len(x))*np.matmul(np.transpose(np.matmul(x_b, self._coef) - y),
                               np.matmul(x_b, self._coef) - y)

        return cost[0][0]

class Log_reg_layer:
    def __init__(self, num_in, num_out, bias_node=True, logistic_func="sigmoid"):
        '''
        Creates a linear regression layer.
        :param num_in: How many input nodes. Does not inlcude biased node
        :param num_out: How many output nodes
        :param bias_node: True if you want to add a biased node. False if not
        :param logistic_func: Which activation function you want to use
        '''
        self._bias = bias_node
        if logistic_func == "sigmoid":
            self._func = self.sigmoid

        if bias_node == True:
            self._coef = self.new_coef(num_in + 1, num_out)
        else:
            self._coef = self.new_coef(num_in, num_out)

    def new_coef(self, dim_one, dim_two):
        '''
        Creates a matrix with random float values.

        :param dim_one: How many rows
        :param dim_two: how many columns
        :return: The matrix
        '''
        matrix = [[] for _ in range(dim_one)]
        for n in range(dim_one):
            for m in range(dim_two):
                matrix[n].append(random.random())
        return matrix

    def _add_bias(self, x):
        '''
        Adds a column of ones to matrix x
        :param x: a matrix
        :return: x with bias nodes
        '''
        x_b = [[] for _ in range(len(x))]
        for n in range(len(x)):
            x_b[n] = [1] + x[n]
        return x_b

    def sigmoid(self, t):
        '''returns the sigmoid function evaluated at t'''
        sig = 1/(1+np.exp(-t))
        return sig


    def feed_for(self, x):
        '''
        Does the feed forward process. Input some data and this function returns predicted value(s) using the coefficients.
        :param x: input matrix
        :return: predicted value(s)
        '''
        if self._bias == True:
            x_b = self._add_bias(x)
        else:
            x_b = x

        return self._func(np.matmul(x_b, self._coef))

    def grad_descent(self, x, y, lamb):
        '''
         Does a gradient descent step, changing the coefficients
         :param x: input data. Data must be in a nested list
         :param y: output data. Data must be in a nested list
         :param lamb: how big the step is
         '''
        if self._bias == True:
            x_b = self._add_bias(x)
        else:
            x_b = x
        y_hat = self._func(np.matmul(x_b, self._coef))

        self._coef -= lamb/len(x)*np.matmul(np.transpose(x_b), y_hat-y)

    def coef_ret(self):
        '''return coefficients'''
        return self._coef

    def cost(self, x, y):
        '''
        Calculates and returns cost
        :param x: input data. Data must be in a nested list
        :param y: output data. Data must be in a nested list
        :return:  the cost
        '''
        if self._bias == True:
            x_b = self._add_bias(x)
        else:
            x_b = x

        y_hat = self._func(np.matmul(x_b, self._coef)) #predicted value
        I = [[1] for _ in range(len(y))]

        cost = 1/len(x_b)*(-np.matmul(np.transpose(y),np.log(y_hat))
                           - np.matmul(np.transpose(np.subtract(I,y)), np.log(np.subtract(I,y_hat))))
        return cost[0][0]



