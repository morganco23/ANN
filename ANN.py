import numpy as np
import math
    

class perceptron_layer:

    # initial weights, af, is_hidden, is_output
    def __init__(self, weights, activation_function, activation_derivative, mu=0.01, is_hidden=False, is_output=False):
        
        self.W = weights                # weights
        self.af = activation_function   # activation function
        self.ad = activation_derivative # activation function derivative
        self.mu = mu                    # learning rate
        self.is_hidden = is_hidden      # is a hidden layer
        self.is_output = is_output      # is an output layer

        self.I = None # inputs
        self.D = None # deltas
        self.O = None # output
        self.E = None # error
        

    def calc_output(self, inputs):
        self.I = inputs # used for delta calculation
        vals = self.af(vals)
        vals = np.dot(self.W, np.insert([1],inputs))
        
        self.O = vals
    
    def calc_deltas(self, error_term):
        
        
        self.D = self.E * self.ad(self.O)


    def update_weights(self):
        self.W = self.W - (self.mu * self.D)



class ANN:

    def __init__(self, num_inputs, num_outputs, num_hidden_layers, hidden_layer_size, activation_function, activation_derivative):

        self.layers = []
        
        # initialize our layers with random weights

        # input layer weights
        self.layers.append(perceptron_layer(np.random.rand(hidden_layer_size, (num_inputs + 1)), activation_function, activation_derivative))

        # hidden layer weights
        for i in range(num_hidden_layers-1):
            self.layers.append(perceptron_layer(np.random.rand(hidden_layer_size, (hidden_layer_size+1)), activation_function, activation_derivative, is_hidden=True))
        
        # output layer weights
        self.layers.append(perceptron_layer(np.random.rand(num_outputs , (hidden_layer_size + 1)), activation_function, activation_derivative, is_output=True))
        
    def print(self):
        for layer in self.layers:
            print(layer.W)
    
    def feed_forward(self, inputs):
        print(inputs)


def main():
    
    def sigmoid(n):
        return 1 / (1 + math.e ** (-n))

    def sigmoid_derivative(n):
        return sigmoid(n) - sigmoid(1-n)

    np.vectorize(sigmoid)
    np.vectorize(sigmoid_derivative)



    my_ann = ANN(5,3,1,4,sigmoid,sigmoid_derivative)

    my_ann.feed_forward(1)

if __name__ == "__main__":
    main()