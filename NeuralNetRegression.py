import numpy as np
import functions # for regression scores
import sys

class NeuralNetRegression():

    def __init__(
            self,
            X_data,
            Y_data,
            hidden_neurons,
            n_categories,
            epochs,
            batch_size,
            eta,
            lmbd,
            activation_f):

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
        self.activation_f  = activation_f


        self.create_biases_and_weights()
        # list of activated values in neurons
        self.activations = []
        self.hidden_layers = len(hidden_neurons)
        #print("The number of hidden layers is: ",self.hidden_layers)

    # Activation functions in Neural Network
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    def sigmoid_prime(self,x):
        return (x*(1-x))
    def RELU(self, x):
        np.clip(x, a_min = 0.0, a_max = sys.float_info.max, out = x)
        #xtemp = max(0,x)
        return x
    def RELUprime(self,x):
        #return 1.0 if x > 0 else 0
        return np.heaviside(x, 0.0)
    def LeakyRELU(self, x):
        # wants z when z>0, alpha*z when z<0
        self.alpha = 0.01
        #print(((x >= 0.0) * x + (x < 0.0) * (self.alpha * x)).shape)
        return (x >= 0.0) * x + (x < 0.0) * (self.alpha * x)
    def LeakyRELUprime(self,x):
        # return 1 if z > 0 else alpha
        #print(x.shape)
        self.alpha = 0.01
        dx = np.ones_like(x)
        dx[x < 0] = self.alpha
        #print(dx.shape)
        return dx
        
    # Initial set up of biases and weights    
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

    # Initial feed forward to start back propagation
    def feed_forward(self):
        # feed-forward for training first layer
        self.z_h = np.matmul(self.X_data, self.hidden_weights[0]) + self.hidden_bias[0]
        if self.activation_f == "sigmoid":
             self.a_h = self.sigmoid(self.z_h)
        if self.activation_f == "RELU":
             self.a_h = self.RELU(self.z_h) 
        if self.activation_f == "LeakyRELU":
             self.a_h = self.LeakyRELU(self.z_h)
        # empty this list
        self.activations = []
        self.activations.append(self.a_h)
        
        # Need these for RELU and LeakyRELU prime??
        self.values = []
        self.values.append(self.z_h)
        
        # checks how many hidden layers we have
        if (len(self.hidden_neurons) > 1):
            #print("We have multiple hidden layers")
            for b, w in zip(self.hidden_bias[1:],self.hidden_weights[1:]):
                
                if self.activation_f == "sigmoid":
                    self.a_h = self.sigmoid(np.matmul(self.a_h,w) + b )

                if self.activation_f == "RELU":
                    self.values.append(np.matmul(self.a_h,w) + b )                
                    self.a_h = self.RELU(np.matmul(self.a_h,w) + b )

                if self.activation_f == "LeakyRELU":
                    self.values.append(np.matmul(self.a_h,w) + b )
                    self.a_h = self.LeakyRELU(np.matmul(self.a_h,w) + b )

                self.activations.append(self.a_h)

        # a_h is calculated at last hidden layer
        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        # do i need an activation function for last layer???
        self.ytilde = self.z_o


    def feed_forward_out(self,X):
        # feed-forward for training first layer
        self.z_h = np.matmul(X, self.hidden_weights[0]) + self.hidden_bias[0]
        if self.activation_f == "sigmoid":
             self.a_h = self.sigmoid(self.z_h)
        if self.activation_f == "RELU":
             self.a_h = self.RELU(self.z_h) 
        if self.activation_f == "LeakyRELU":
             self.a_h = self.LeakyRELU(self.z_h)
      
        # checks how many hidden layers we have
        if (len(self.hidden_neurons) > 1):
            #print("We have multiple hidden layers")
            for b, w in zip(self.hidden_bias[1:],self.hidden_weights[1:]):

                if self.activation_f == "sigmoid":
                    self.a_h = self.sigmoid(np.matmul(self.a_h,w) + b )

                if self.activation_f == "RELU":
                    self.a_h = self.RELU(np.matmul(self.a_h,w) + b ) 

                if self.activation_f == "LeakyRELU":
                    self.a_h = self.LeakyRELU(np.matmul(self.a_h,w) + b )

        # a_h is calculated at last hidden layer
        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        # do i need an activation function for last layer???
        self.ytilde = self.z_o
        return self.ytilde        
    
    def predict(self, X,y_test,X_train,y_train,printMe):
        ytilde = self.feed_forward_out(X)
        ytilde = ytilde.reshape(-1) #ravel()

        if (printMe==True):
            print("\nValues calculated for test data:")
            print("The NeuralNetwork MSE score is: {:.4f}".format(functions.MSE(y_test,ytilde)))
            print("The NeuralNetwork R2 score is: {:.4f}".format(functions.R2(y_test,ytilde)))       
            
            print("\nValues calculated for training data:")
            ytilde = self.feed_forward_out(X_train)
            ytilde = ytilde.reshape(-1)         
            print("The NeuralNetwork MSE score is: {:.4f}".format(functions.MSE(y_train,ytilde)))
            print("The NeuralNetwork R2 score is: {:.4f}".format(functions.R2(y_train,ytilde)))

        if (printMe==False):
            ytilde = self.feed_forward_out(X)
            ytilde = ytilde.reshape(-1) #ravel()        
            MSE_test=functions.MSE(y_test,ytilde)
            R2_test=functions.R2(y_test,ytilde)
            return MSE_test,R2_test
        

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def backpropagation(self):
        #self.ytilde = self.ytilde.reshape(-1)
        self.Y_data.shape = (self.Y_data.shape[0],1)
        error_output = self.ytilde - self.Y_data

        #print("ytilde shape is: ",self.ytilde.shape)
        #print("ydata shape is: ",self.Y_data.shape)
        #print("error shape is: ",error_output.shape)
        
        # self.a_h is last value calculated so ok..
        errors = []

        if self.activation_f == "sigmoid":
            error_hidden = np.matmul(error_output, self.output_weights.T) * self.sigmoid_prime(self.a_h) #self.a_h * (1 - self.a_h)

        if self.activation_f == "RELU":
            error_hidden = np.matmul(error_output, self.output_weights.T) * self.RELUprime(self.values[-1])

        if self.activation_f == "LeakyRELU":
            error_hidden = np.matmul(error_output, self.output_weights.T) * self.LeakyRELUprime(self.values[-1])

        errors.append(error_hidden)

        for i in range(self.hidden_layers - 1)[::-1]:
            #print("Calculating errors layer: ",i)
            #
            # Is the wrong error value used here? NO

            if self.activation_f == "sigmoid":
                error_hidden = np.matmul(errors[0], self.hidden_weights[i+1].T) \
                    * self.sigmoid_prime(self.activations[i]) #* self.activations[i] * (1 - self.activations[i])

            if self.activation_f == "RELU":
                error_hidden = np.matmul(errors[0], self.hidden_weights[i+1].T) \
                    * self.RELUprime(self.values[i]) #* self.activations[i] * (1 - self.activations[i])

            if self.activation_f == "LeakyRELU":
                error_hidden = np.matmul(errors[0], self.hidden_weights[i+1].T) \
                    * self.LeakyRELUprime(self.activations[i]) #* self.activations[i] * (1 - self.activations[i])
 
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
            for i in range(self.hidden_layers):
                gradients_hw[i] += self.lmbd * gradients_hw[i]

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient

        for i in range(self.hidden_layers):
            self.hidden_weights[i] -= self.eta* gradients_hw[i]
            self.hidden_bias[i] -= self.eta* gradients_hb[i]


    def train(self):
        #print("\nTraining Neural Network for Regression")
        print("Chosen activation function is: ",self.activation_f)         

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
            