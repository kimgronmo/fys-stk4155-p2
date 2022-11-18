import numpy as np
import sys

class NeuralNetClassification():

    def __init__(
            self,
            X_data,
            Y_data,
            hidden_neurons,
            n_categories,
            epochs,
            batch_size,
            eta,
            lmbd):

        self.X_data_full = X_data
        self.Y_data_full = Y_data
        
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.hidden_neurons = hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()
        # list of activated values in neurons
        self.activations = []
        self.hidden_layers = len(hidden_neurons)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    def RELU(self, x):
        np.clip(x, a_min = 0.0, a_max = sys.float_info.max, out = x)
        #xtemp = max(0,x)
        return x

    def create_biases_and_weights(self):
        # creates biases and weights
        #print("\nCreates biases and weights")
        # print to see if created correctly
        input = np.array([self.n_features])
        for i in self.hidden_neurons:
            input = np.append(input,i)
        
        self.hidden_weights = []
        self.hidden_weights.append (np.array(np.random.randn(self.n_features,self.hidden_neurons[0])))
        if len(self.hidden_neurons) > 1:
            for i in range(len(self.hidden_neurons)-1):
                temparray = np.array(np.random.randn(input[i+1],input[i+2]))
                self.hidden_weights.append(temparray)
                
        self.hidden_bias = [(np.zeros(x) + 0.01) for x in self.hidden_neurons[0:]]
        self.output_weights = np.random.randn(self.hidden_neurons[-1],self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01


    def feed_forward(self):
        # feed-forward for training first layer
        self.z_h = np.matmul(self.X_data, self.hidden_weights[0]) + self.hidden_bias[0]
        self.a_h = self.sigmoid(self.z_h)
        #self.a_h = self.RELU(self.z_h)        
        # empty this list
        self.activations = []
        self.activations.append(self.a_h)

        # checks how many hidden layers we have
        if (len(self.hidden_neurons) > 1):
            #print("We have multiple hidden layers")
            for b, w in zip(self.hidden_bias[1:],self.hidden_weights[1:]):
                self.a_h = self.sigmoid(np.matmul(self.a_h,w) + b )
                #self.a_h = self.RELU(np.matmul(self.a_h,w) + b )                
                self.activations.append(self.a_h)

        # a_h is calculated at last hidden layer
        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        # softmax function for probabilities
        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True) 

    def feed_forward_out(self,X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights[0]) + self.hidden_bias[0]
        a_h = self.sigmoid(z_h)
        #a_h = self.RELU(z_h)        

        if (len(self.hidden_neurons) > 1):
            for b, w in zip(self.hidden_bias[1:],self.hidden_weights[1:]):
                a_h = self.sigmoid(np.matmul(a_h,w) + b)
                #a_h = self.RELU(np.matmul(a_h,w) + b)                
        
        # a_h is the last activated values in last hidden layer
        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities        
        

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data
        
        # self.a_h is last value calculated so ok..
        errors = []
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)
        errors.append(error_hidden)

        for i in range(self.hidden_layers - 1)[::-1]:
            error_hidden = np.matmul(errors[0], self.hidden_weights[i+1].T) \
                * self.activations[i] * (1 - self.activations[i])
            errors.insert(0,error_hidden)

        # uses last a_h calculated is ok as is..
        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        gradients_hw = []
        gradients_hb = []
        for i in range(self.hidden_layers):
            if i == 0:
                self.hidden_weights_gradient = np.matmul(self.X_data.T, errors[0])
                self.hidden_bias_gradient = np.sum(errors[0], axis=0)
                gradients_hw.append(self.hidden_weights_gradient)
                gradients_hb.append(self.hidden_bias_gradient)
            else:
                # should use previous self activation.
                self.hidden_weights_gradient = np.matmul(self.activations[i-1].T, errors[i])
                self.hidden_bias_gradient = np.sum(errors[i], axis=0)
                gradients_hw.append(self.hidden_weights_gradient)
                gradients_hb.append(self.hidden_bias_gradient)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            #self.hidden_weights_gradient += self.lmbd * self.hidden_weights[0]
            for i in range(self.hidden_layers):
                gradients_hw[i] += self.lmbd * gradients_hw[i]

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient

        for i in range(self.hidden_layers):
            self.hidden_weights[i] -= self.eta* gradients_hw[i]
            self.hidden_bias[i] -= self.eta* gradients_hb[i]

    def train(self):
        #print("\nTraining Neural Network for Number classification")    

        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                # arange to do this?
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()
            # print how well we are doing so far...
            