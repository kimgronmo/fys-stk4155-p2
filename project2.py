import numpy as np
import sklearn.linear_model as skl
import math
from sklearn.neural_network import MLPClassifier as mlpc

import autograd.numpy as np
from autograd import grad

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import helper functions
# from project 1
import functions
import Printerfunctions

# imports neural network
import NeuralNetClassification # for image classification
import NeuralNetRegression # for regression analysis

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets # for mnist data if needed

import warnings

# Generates dataset based upon the Franke function
def generateDataset(seed,n,N):
    print("Generating the FrankeFunction Dataset:",end="")

    # Using a seed to ensure that the random numbers are the same everytime we run
    # the code. Useful to debug and check our code.
    np.random.seed(seed)

    # Basic information for the FrankeFunction
    # The degree of the polynomial (number of features) is given by
    n = n
    # the number of datapoints
    N = N

    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    # Remember to add noise to function 
    z = functions.FrankeFunction(x, y) + 0.01*np.random.rand(N)

    X = functions.create_X(x, y, n=n)

    # split in training and test data
    # assumes 75% of data is training and 25% is test data
    X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

    # scaling the data
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)
    
    y_train -= np.mean(y_train)
    y_test -= np.mean(y_test)
    
    
    print(" Done\n")
    return X_train,X_test,y_train,y_test

# Generate training and test data for the MNIST dataset
def generateDatasetImages():
    #import matplotlib.pyplot as plt
    print("\nGenerating dataset from MNIST \n")
    # ensure the same random numbers appear every time
    np.random.seed(0)

    # download MNIST dataset
    digits = datasets.load_digits()

    # define inputs and labels
    inputs = digits.images
    labels = digits.target

    n_inputs = len(inputs)
    inputs = inputs.reshape(n_inputs, -1)

    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
                                                    test_size=test_size)

    # to categorical turns our integer vector into a onehot representation
    #    one-hot in numpy
    Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train), to_categorical_numpy(Y_test)  
    return X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot

# Generates a dataset using Wisconsin Breast Cancer Data
def generateDatasetWBC():
    print("\nGenerating dataset from Wisconsin Breast Cancer Data\n")
    from sklearn.datasets import load_breast_cancer
    cancer=load_breast_cancer()      #Download breast cancer dataset

    inputs=cancer.data       
    outputs=cancer.target    
    labels=cancer.feature_names[0:30]

    """
    print('The content of the breast cancer dataset is:')      #Print information about the datasets
    print(labels)
    print('-------------------------')
    print("inputs =  " + str(inputs.shape))
    print("outputs =  " + str(outputs.shape))
    print("labels =  "+ str(labels.shape))
    """

    x=inputs      #Reassign the Feature and Label matrices to other variables
    y=outputs    
    # Generate training and testing datasets

    #Select features relevant to classification (texture,perimeter,compactness and symmetry) 
    #and add to input matrix

    temp1=np.reshape(x[:,1],(len(x[:,1]),1))
    temp2=np.reshape(x[:,2],(len(x[:,2]),1))
    X=np.hstack((temp1,temp2))      
    temp=np.reshape(x[:,5],(len(x[:,5]),1))
    X=np.hstack((X,temp))       
    temp=np.reshape(x[:,8],(len(x[:,8]),1))
    X=np.hstack((X,temp))       

    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)   #Split datasets into training and testing
    
    #Convert labels to categorical when using categorical cross entropy
    Y_train_onehot=to_categorical_numpy(Y_train)     
    Y_test_onehot=to_categorical_numpy(Y_test)

    del temp1,temp2,temp
    
    epochs = 1000
    batch_size = 100
    eta = 0.01
    lmbd = 0.01

    n_categories = 2
    #hidden_neurons = [30,20,10]
    hidden_neurons = [50,50]

    dnn2 = NeuralNetClassification.NeuralNetClassification(X_train, Y_train_onehot,eta=eta,lmbd=lmbd,epochs=epochs,batch_size=batch_size,
                    hidden_neurons=hidden_neurons,n_categories=n_categories)
    dnn2.train()
    test_predict = dnn2.predict(X_test)

    # accuracy score from scikit library
    training_predict = dnn2.predict(X_train)
    
    test_predict=dnn2.predict(X_test)#_probabilities(X_test)

    print("Accuracy score on training set: {:.4f}".format(accuracy_score(Y_train, training_predict)))    
    print("Accuracy score on test set: {:.4f}".format(accuracy_score(Y_test, test_predict)))    

    sklearn_classifier=mlpc(hidden_layer_sizes=(50,50),learning_rate_init=eta)
    sklearn_classifier.fit(X_train,Y_train_onehot)
    training_predict = sklearn_classifier.predict(X_train)
    
    test_predict=sklearn_classifier.predict(X_test)#_probabilities(X_test)    

    training_predict=functions.to_integer_vector(training_predict)
    test_predict = functions.to_integer_vector(test_predict)

    print("\nFor sklearn mlpclassifier:")
    print("Accuracy score on training set: {:.4f}".format(accuracy_score(Y_train, training_predict)))    
    print("Accuracy score on test set: {:.4f}".format(accuracy_score(Y_test, test_predict)))   


    
    return X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot

# Returns categories as a onehot vector    
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    return onehot_vector    

# Finds optimal values for stochastic gradient descent methods with optional tuning    
# Ridge/ols,beta/theta,GD/SGD,moments,printing,tuning
def analyse_gradient_descent(X_train,y_train,X_test,y_test,method,beta,descent,batch_sizes,moments,printing,tuning):
    best_R2=-1000.0
    best_momentum=-1.0
    best_eta=-1.0
    best_lmbd=-1.0    
    best_batch_size=-1.0
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    epochs=1000
    beta_original=beta
    
    if (method=="OLS"):
        lmbd_vals=[0.0]
    
    if (descent=="SGD"):
        for l in range(len(batch_sizes)):
            for k in range(len(lmbd_vals)):     
                for i in range(len(eta_vals)):
                    for j in range(len(moments)):
                        eta=eta_vals[i]
                        momentum=moments[j]
                        lmbd=lmbd_vals[k]
                        mini_batch_size=batch_sizes[l]
                        beta_o=np.copy(beta_original)
                        #print(beta_o)
                        R2_score=functions.SGD(X_train,y_train,X_test,y_test,epochs,momentum,mini_batch_size,eta,beta_o,lmbd,method,printing,tuning)
                        #print(R2_score)
                        if (math.isnan(R2_score))==False:
                            if (R2_score > best_R2) & (R2_score <= 1.0):
                                #print("R2 score: ",R2_score)
                                best_R2=R2_score
                                best_momentum=momentum
                                best_eta=eta
                                best_lmbd=lmbd_vals[k]
                                best_batch_size=batch_sizes[l]
    return best_R2,best_momentum,best_eta,best_batch_size,best_lmbd


# Part a of the project
def part_a(X_train,X_test,y_train,y_test,N):
    print("Starting Project 2: part a")
    print("")

    beta_original = np.random.randn(X_train[0].size)
    theta_original = np.random.randn(X_train[0].size)
    
    beta = np.copy(beta_original) # test all methods with same beta
    theta = np.copy(theta_original)
    epochs = 1000
    eta_vals = np.logspace(-5, 1, 7)    
    lmbd=0.0
    
    print("\n#### OLS Regression with GD ####")    
    print("Calculating OLS with GD and no momentum")
    momentum=0.0
    R2_scores=np.zeros(len(eta_vals))
    for i in range(len(eta_vals)):
        eta=eta_vals[i]
        R2_scores[i]=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,beta,"OLS",lmbd,True,"None")
        #print(R2_scores[i])
    best_R2, index = functions.find_largest_score(R2_scores)
    print("Max R2 score: ",best_R2," for eta value: ",eta_vals[index])

    moments = np.linspace(0.1,1.0,4)
    print("\nCalculating OLS with GD and momentum")
    beta = np.copy(beta_original) # test all methods with same beta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    for i in range(len(eta_vals)):
        for j in range(len(moments)):
            eta=eta_vals[i]
            momentum=moments[j]
            R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,beta,"OLS",lmbd,True,"None")
            #print(R2_score)
            if (math.isnan(R2_score))==False:
                if (R2_score > best_R2) & (R2_score <= 1.0):
                    best_R2=R2_score
                    best_momentum=momentum
                    best_eta=eta
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum)

    print("\nCalculating OLS with GD, time decay rate tuning and no momentum")
    moments=[0.0]
    beta = np.copy(beta_original) # test all methods with same beta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    for i in range(len(eta_vals)):
        for j in range(len(moments)):
            eta=eta_vals[i]
            momentum=moments[j]
            R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,beta,"OLS",lmbd,True,"tdr")
            #print(R2_score)
            if (math.isnan(R2_score))==False:
                if (R2_score > best_R2) & (R2_score <= 1.0):
                    best_R2=R2_score
                    best_momentum=momentum
                    best_eta=eta
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: time decay rate")

    print("\nCalculating OLS with GD, time decay rate tuning and momentum")
    moments=np.linspace(0.1,1.0,4)
    beta = np.copy(beta_original) # test all methods with same beta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    for i in range(len(eta_vals)):
        for j in range(len(moments)):
            eta=eta_vals[i]
            momentum=moments[j]
            R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,beta,"OLS",lmbd,True,"tdr")
            #print(R2_score)
            if (math.isnan(R2_score))==False:
                if (R2_score > best_R2) & (R2_score <= 1.0):
                    best_R2=R2_score
                    best_momentum=momentum
                    best_eta=eta
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: time decay rate")    
    
    print("\nCalculating OLS with GD, Adagrad tuning and no momentum")
    moments=[0.0]
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    for i in range(len(eta_vals)):
        for j in range(len(moments)):
            eta=eta_vals[i]
            momentum=moments[j]
            R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,theta,"OLS",lmbd,True,"Adagrad")
            #print(R2_score)
            if (math.isnan(R2_score))==False:
                if (R2_score > best_R2) & (R2_score <= 1.0):
                    best_R2=R2_score
                    best_momentum=momentum
                    best_eta=eta
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: Adagrad")    

    print("\nCalculating OLS with GD, Adagrad tuning and momentum")
    moments=np.linspace(0.1,1.0,4)
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    for i in range(len(eta_vals)):
        for j in range(len(moments)):
            eta=eta_vals[i]
            momentum=moments[j]
            R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,theta,"OLS",lmbd,True,"Adagrad")
            #print(R2_score)
            if (math.isnan(R2_score))==False:
                if (R2_score > best_R2) & (R2_score <= 1.0):
                    best_R2=R2_score
                    best_momentum=momentum
                    best_eta=eta
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: Adagrad")      

    print("\nCalculating OLS with GD, RMSprop tuning and no momentum")
    moments=[0.0]
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    for i in range(len(eta_vals)):
        for j in range(len(moments)):
            eta=eta_vals[i]
            momentum=moments[j]
            R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,theta,"OLS",lmbd,True,"RMSprop")
            #print(R2_score)
            if (math.isnan(R2_score))==False:
                if (R2_score > best_R2) & (R2_score <= 1.0):
                    best_R2=R2_score
                    best_momentum=momentum
                    best_eta=eta
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: RMSprop") 

    print("\nCalculating OLS with GD, RMSprop tuning and momentum")
    moments=np.linspace(0.1,1.0,4)
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    for i in range(len(eta_vals)):
        for j in range(len(moments)):
            eta=eta_vals[i]
            momentum=moments[j]
            R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,theta,"OLS",lmbd,True,"RMSprop")
            #print(R2_score)
            if (math.isnan(R2_score))==False:
                if (R2_score > best_R2) & (R2_score <= 1.0):
                    best_R2=R2_score
                    best_momentum=momentum
                    best_eta=eta
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: RMSprop")  

    print("\nCalculating OLS with GD, Adam tuning and no momentum")
    moments=[0.0]
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    for i in range(len(eta_vals)):
        for j in range(len(moments)):
            eta=eta_vals[i]
            momentum=moments[j]
            R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,theta,"OLS",lmbd,True,"Adam")
            #print(R2_score)
            if (math.isnan(R2_score))==False:
                if (R2_score > best_R2) & (R2_score <= 1.0):
                    best_R2=R2_score
                    best_momentum=momentum
                    best_eta=eta
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: Adam") 

    print("\nCalculating OLS with GD, Adam tuning and momentum")
    moments=np.linspace(0.1,1.0,4)
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    for i in range(len(eta_vals)):
        for j in range(len(moments)):
            eta=eta_vals[i]
            momentum=moments[j]
            R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,theta,"OLS",lmbd,True,"Adam")
            #print(R2_score)
            if (math.isnan(R2_score))==False:
                if (R2_score > best_R2) & (R2_score <= 1.0):
                    best_R2=R2_score
                    best_momentum=momentum
                    best_eta=eta
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: Adam")  
    
    print("\n#### Ridge Regression with GD ####")
    #### Now for Ridge regression ####
    
    lmbd_vals = np.logspace(-5, 1, 7)
    
    print("\nCalculating Ridge with GD and no momentum")
    momentum=0.0
    best_R2=-1.0
    best_lmbd=-1.0
    for k in range(len(lmbd_vals)):
        for i in range(len(eta_vals)):
            eta=eta_vals[i]
            lmbd=lmbd_vals[k]
            R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,beta,"Ridge",lmbd,True,"None")
            if (math.isnan(R2_score))==False:
                if (R2_score > best_R2) & (R2_score <= 1.0):
                    best_R2=R2_score
                    best_momentum=momentum
                    best_eta=eta
                    best_lmbd=lmbd_vals[k]        
    
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," momentum: ",best_momentum," with lambda: ",best_lmbd)

    
    moments = np.linspace(0.1,1.0,4)
    print("\nCalculating Ridge with GD and momentum")
    beta = np.copy(beta_original) # test all methods with same beta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    best_lmbd=-1.0
    for k in range(len(lmbd_vals)):
        for i in range(len(eta_vals)):
            for j in range(len(moments)):
                eta=eta_vals[i]
                momentum=moments[j]
                lmbd=lmbd_vals[k]                
                R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,beta,"Ridge",lmbd,True,"None")
                #print(R2_score)
                if (math.isnan(R2_score))==False:
                    if (R2_score > best_R2) & (R2_score <= 1.0):
                        best_R2=R2_score
                        best_momentum=momentum
                        best_eta=eta
                        best_lmbd=lmbd_vals[k]
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," with lambda: ",best_lmbd)

    print("\nCalculating Ridge with GD, time decay rate tuning and no momentum")
    moments=[0.0]
    beta = np.copy(beta_original) # test all methods with same beta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    best_lmbd=-1.0
    for k in range(len(lmbd_vals)):
        for i in range(len(eta_vals)):
            for j in range(len(moments)):
                eta=eta_vals[i]
                momentum=moments[j]
                lmbd=lmbd_vals[k]                  
                R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,beta,"Ridge",lmbd,True,"tdr")
                #print(R2_score)
                if (math.isnan(R2_score))==False:
                    if (R2_score > best_R2) & (R2_score <= 1.0):
                        best_R2=R2_score
                        best_momentum=momentum
                        best_eta=eta
                        best_lmbd=lmbd_vals[k]
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: time decay rate"," with lambda: ",best_lmbd)

    print("\nCalculating Ridge with GD, time decay rate tuning and momentum")
    moments=np.linspace(0.1,1.0,4)
    beta = np.copy(beta_original) # test all methods with same beta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    best_lmbd=-1.0
    for k in range(len(lmbd_vals)):
        for i in range(len(eta_vals)):
            for j in range(len(moments)):
                eta=eta_vals[i]
                momentum=moments[j]
                lmbd=lmbd_vals[k]                  
                R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,beta,"Ridge",lmbd,True,"tdr")
                #print(R2_score)
                if (math.isnan(R2_score))==False:
                    if (R2_score > best_R2) & (R2_score <= 1.0):
                        best_R2=R2_score
                        best_momentum=momentum
                        best_eta=eta
                        best_lmbd=lmbd_vals[k]
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: time decay rate"," with lambda: ",best_lmbd)    
    

    
    print("\nCalculating Ridge with GD, Adagrad tuning and no momentum")
    moments=[0.0]
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    best_lmbd=-1.0
    for k in range(len(lmbd_vals)):    
        for i in range(len(eta_vals)):
            for j in range(len(moments)):
                eta=eta_vals[i]
                momentum=moments[j]
                lmbd=lmbd_vals[k]                  
                R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,theta,"Ridge",lmbd,True,"Adagrad")
                #print(R2_score)
                if (math.isnan(R2_score))==False:
                    if (R2_score > best_R2) & (R2_score <= 1.0):
                        best_R2=R2_score
                        best_momentum=momentum
                        best_eta=eta
                        best_lmbd=lmbd_vals[k]
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: Adagrad"," with lambda: ",best_lmbd)      

    print("\nCalculating Ridge with GD, Adagrad tuning and momentum")
    moments=np.linspace(0.1,1.0,4)
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    best_lmbd=-1.0    
    for k in range(len(lmbd_vals)):     
        for i in range(len(eta_vals)):
            for j in range(len(moments)):
                eta=eta_vals[i]
                momentum=moments[j]
                lmbd=lmbd_vals[k]                  
                R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,theta,"Ridge",lmbd,True,"Adagrad")
                #print(R2_score)
                if (math.isnan(R2_score))==False:
                    if (R2_score > best_R2) & (R2_score <= 1.0):
                        best_R2=R2_score
                        best_momentum=momentum
                        best_eta=eta
                        best_lmbd=lmbd_vals[k]                    
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: Adagrad"," with lambda: ",best_lmbd)      

    print("\nCalculating Ridge with GD, RMSprop tuning and no momentum")
    moments=[0.0]
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    best_lmbd=-1.0    
    for k in range(len(lmbd_vals)):     
        for i in range(len(eta_vals)):
            for j in range(len(moments)):
                eta=eta_vals[i]
                momentum=moments[j]
                lmbd=lmbd_vals[k]                  
                R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,theta,"Ridge",lmbd,True,"RMSprop")
                #print(R2_score)
                if (math.isnan(R2_score))==False:
                    if (R2_score > best_R2) & (R2_score <= 1.0):
                        best_R2=R2_score
                        best_momentum=momentum
                        best_eta=eta
                        best_lmbd=lmbd_vals[k]
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: RMSprop"," with lambda: ",best_lmbd) 

    print("\nCalculating Ridge with GD, RMSprop tuning and momentum")
    moments=np.linspace(0.1,1.0,4)
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    best_lmbd=-1.0  
    for k in range(len(lmbd_vals)):     
        for i in range(len(eta_vals)):
            for j in range(len(moments)):
                eta=eta_vals[i]
                momentum=moments[j]
                lmbd=lmbd_vals[k]                  
                R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,theta,"Ridge",lmbd,True,"RMSprop")
                #print(R2_score)
                if (math.isnan(R2_score))==False:
                    if (R2_score > best_R2) & (R2_score <= 1.0):
                        best_R2=R2_score
                        best_momentum=momentum
                        best_eta=eta
                        best_lmbd=lmbd_vals[k]
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: RMSprop"," with lambda: ",best_lmbd)  

    print("\nCalculating Ridge with GD, Adam tuning and no momentum")
    moments=[0.0]
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    best_lmbd=-1.0
    for k in range(len(lmbd_vals)):     
        for i in range(len(eta_vals)):
            for j in range(len(moments)):
                eta=eta_vals[i]
                momentum=moments[j]
                lmbd=lmbd_vals[k]                  
                R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,theta,"Ridge",lmbd,True,"Adam")
                #print(R2_score)
                if (math.isnan(R2_score))==False:
                    if (R2_score > best_R2) & (R2_score <= 1.0):
                        best_R2=R2_score
                        best_momentum=momentum
                        best_eta=eta
                        best_lmbd=lmbd_vals[k]
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: Adam"," with lambda: ",best_lmbd) 

    print("\nCalculating Ridge with GD, Adam tuning and momentum")
    moments=np.linspace(0.1,1.0,4)
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2=-1.0
    best_momentum=-1.0
    best_eta=-1.0
    best_lmbd=-1.0    
    for k in range(len(lmbd_vals)):     
        for i in range(len(eta_vals)):
            for j in range(len(moments)):
                eta=eta_vals[i]
                momentum=moments[j]
                lmbd=lmbd_vals[k]                  
                R2_score=functions.GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,theta,"Ridge",lmbd,True,"Adam")
                #print(R2_score)
                if (math.isnan(R2_score))==False:
                    if (R2_score > best_R2) & (R2_score <= 1.0):
                        best_R2=R2_score
                        best_momentum=momentum
                        best_eta=eta
                        best_lmbd=lmbd_vals[k]
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with momentum: ",momentum," for tuning: Adam"," with lambda: ",best_lmbd)
    
    print("\n#### OLS Regression with SGD ####")
    #### For SGD with OLS and Ridge ###
    batch_sizes=[32,64]
    
    
    print("\nCalculating OLS with SGD and no momentum")
    momentum=[0.0]
    beta = np.copy(beta_original)    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "OLS",beta,"SGD",batch_sizes,momentum,True,"None")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)

    print("\nCalculating OLS with SGD and momentum")
    momentum=np.linspace(0.1,1.0,4)
    beta = np.copy(beta_original)    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "OLS",beta,"SGD",batch_sizes,momentum,True,"None")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)
    

    print("\nCalculating OLS with SGD, time decay rate tuning and no momentum")
    momentum=[0.0]
    beta = np.copy(beta_original) # test all methods with same beta    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "OLS",beta,"SGD",batch_sizes,momentum,True,"tdr")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)

    print("\nCalculating OLS with SGD, time decay rate tuning and momentum")
    momentum=np.linspace(0.1,1.0,4)
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "OLS",beta,"SGD",batch_sizes,momentum,True,"tdr")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)
    


    
    print("\nCalculating OLS with SGD, Adagrad tuning and no momentum")
    momentum=[0]
    theta = np.copy(theta_original) # test all methods with same theta     
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "OLS",theta,"SGD",batch_sizes,momentum,True,"Adagrad")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)   

    print("\nCalculating OLS with SGD, Adagrad tuning and momentum")
    momentum=np.linspace(0.1,1.0,4)
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "OLS",theta,"SGD",batch_sizes,momentum,True,"Adagrad")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)       
    

    
    print("\nCalculating OLS with SGD, RMSprop tuning and no momentum")
    momentum=[0.0]
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "OLS",theta,"SGD",batch_sizes,momentum,True,"RMSprop")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)   

    print("\nCalculating OLS with SGD, RMSprop tuning and momentum")
    momentum=np.linspace(0.1,1.0,4)
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "OLS",theta,"SGD",batch_sizes,momentum,True,"RMSprop")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)    

    print("\nCalculating OLS with SGD, Adam tuning and no momentum")
    momentum=[0.0]
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "OLS",theta,"SGD",batch_sizes,momentum,True,"Adam")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)   

    print("\nCalculating OLS with SGD, Adam tuning and momentum")
    momentum=np.linspace(0.1,1.0,4)
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "OLS",theta,"SGD",batch_sizes,momentum,True,"Adam")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)   
    
    print("\n#### Ridge Regression with SGD ####")    
    #### Now the same for Ridge regression ####

    print("\nCalculating Ridge with SGD and no momentum")
    momentum=[0.0]
    beta = np.copy(beta_original)    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "Ridge",beta,"SGD",batch_sizes,momentum,True,"None")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)

    print("\nCalculating Ridge with SGD and momentum")
    momentum=np.linspace(0.1,1.0,4)
    beta = np.copy(beta_original)    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "Ridge",beta,"SGD",batch_sizes,momentum,True,"None")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)
    

    print("\nCalculating Ridge with SGD, time decay rate tuning and no momentum")
    momentum=[0.0]
    beta = np.copy(beta_original) # test all methods with same beta    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "Ridge",beta,"SGD",batch_sizes,momentum,True,"tdr")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)

    print("\nCalculating Ridge with SGD, time decay rate tuning and momentum")
    momentum=np.linspace(0.1,1.0,4)
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "Ridge",beta,"SGD",batch_sizes,momentum,True,"tdr")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)
    


    
    print("\nCalculating Ridge with SGD, Adagrad tuning and no momentum")
    momentum=[0]
    theta = np.copy(theta_original) # test all methods with same theta     
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "Ridge",theta,"SGD",batch_sizes,momentum,True,"Adagrad")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)   

    print("\nCalculating Ridge with SGD, Adagrad tuning and momentum")
    momentum=np.linspace(0.1,1.0,4)
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "Ridge",theta,"SGD",batch_sizes,momentum,True,"Adagrad")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)       
    

    
    print("\nCalculating Ridge with SGD, RMSprop tuning and no momentum")
    momentum=[0.0]
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "Ridge",theta,"SGD",batch_sizes,momentum,True,"RMSprop")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)   

    print("\nCalculating Ridge with SGD, RMSprop tuning and momentum")
    momentum=np.linspace(0.1,1.0,4)
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "Ridge",theta,"SGD",batch_sizes,momentum,True,"RMSprop")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)    

    print("\nCalculating Ridge with SGD, Adam tuning and no momentum")
    momentum=[0.0]
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "Ridge",theta,"SGD",batch_sizes,momentum,True,"Adam")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)   

    print("\nCalculating Ridge with SGD, Adam tuning and momentum")
    momentum=np.linspace(0.1,1.0,4)
    theta = np.copy(theta_original) # test all methods with same theta    
    best_R2,best_momentum,best_eta,best_batch_size,best_lmbd = analyse_gradient_descent(X_train,y_train,X_test,y_test, \
        "Ridge",theta,"SGD",batch_sizes,momentum,True,"Adam")
    print("Max R2 score: ",best_R2," for eta value: ",best_eta," with lambda: ", \
        best_lmbd," mini batch size: ",best_batch_size," and momentum: ",best_momentum)  

# Part b and c of the project
def part_b_and_c(X_train,X_test,y_train,y_test):
    print("\n""\n")

    print("Starting Project 2: part b and c")
    print("\n")

    epochs = 10000
    batch_size = 100

    # Depending on the number of hidden layers and neurons
    # RELU and LeakyRELU seem to be susceptible to overflow errors when 
    # calculating the gradient. Varies with #epochs and eta
    # Not sure if it is something wrong with my code that causes it or
    # if they just need the right parameters to function correctly
    # Left hidden neurons as two layers below to show that it works
    # for multiple layers.

    # multiple hidden layers are very time consuming for RELU and LeakyRELU
    # uses 1 layer instead
    #hidden_neurons = [30,28]
    
    # Grid search is time consuming. Using fewer neurons
    hidden_neurons = [20]
    n_categories = 1

    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)

    DNN_numpy = np.zeros((len(eta_vals),len(lmbd_vals)),dtype=object)
    import NeuralNetRegression
    
    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            print("Training neural network for regression for eta: ",eta," and lambda: ",lmbd)
            dnn = NeuralNetRegression.NeuralNetRegression(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    hidden_neurons=hidden_neurons, n_categories=n_categories,activation_f="sigmoid")
            dnn.train()
            DNN_numpy[i][j] = dnn

    MSE_scores_test = np.zeros((len(eta_vals),len(lmbd_vals)))
    R2_scores_test = np.zeros((len(eta_vals),len(lmbd_vals)))

    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            dnn = DNN_numpy[i][j]
            MSE_test,R2_test = dnn.predict(X_test,y_test,X_train,y_train,printMe=False)
            MSE_scores_test[i][j] = MSE_test
            R2_scores_test[i][j] = R2_test

    printer = Printerfunctions.Printerfunctions()
    printer.partB1(R2_scores_test)
    printer.partB2(MSE_scores_test)
    

    eta = 0.0001
    lmbd = 0.01

    ######## Sigmoid
    print("Training NN using sigmoid activation function")
    dnnRegression = NeuralNetRegression.NeuralNetRegression(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    hidden_neurons=hidden_neurons, n_categories=n_categories,activation_f="sigmoid")
    dnnRegression.train()
    dnnRegression.predict(X_test,y_test,X_train,y_train,printMe=True)

    ######## RELU
    print("Training NN using RELU activation function")    
    dnnRegressionRELU = NeuralNetRegression.NeuralNetRegression(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    hidden_neurons=hidden_neurons, n_categories=n_categories,activation_f="RELU")
    dnnRegressionRELU.train()
    dnnRegressionRELU.predict(X_test,y_test,X_train,y_train,printMe=True)

    ######## LeakyRELU
    print("Training NN using LeakyRELU activation function")     
    dnnRegressionLeakyRELU = NeuralNetRegression.NeuralNetRegression(X_train, y_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    hidden_neurons=hidden_neurons, n_categories=n_categories,activation_f="LeakyRELU")
    dnnRegressionLeakyRELU.train()
    dnnRegressionLeakyRELU.predict(X_test,y_test,X_train,y_train,printMe=True)

# Part d of the project
def part_d(X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot,dataset):
    print("\nStarting Project 2: part d")

    epochs = 1000
    batch_size = 100
    if (dataset=="MNIST"):
        n_categories=10
    if (dataset=="WBC"):
        n_categories=2
    hidden_neurons = [30,20,10]    

    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)

    DNN_numpy = np.zeros((len(eta_vals),len(lmbd_vals)),dtype=object)
    import NeuralNetClassification
    
    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            #print(eta," ",lmbd)
            dnn = NeuralNetClassification.NeuralNetClassification(X_train,Y_train_onehot,eta=eta,lmbd=lmbd,epochs=epochs,batch_size=batch_size,
                hidden_neurons=hidden_neurons,n_categories=n_categories)
            dnn.train()
            DNN_numpy[i][j] = dnn
            #test_predict = dnn.predict(X_test)

    accuracy_scores_train = np.zeros((len(eta_vals),len(lmbd_vals)))
    accuracy_scores_test = np.zeros((len(eta_vals),len(lmbd_vals)))

    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            dnn = DNN_numpy[i][j]
            train_pred = dnn.predict(X_train) 
            test_pred = dnn.predict(X_test)
            accuracy_scores_train[i][j] = accuracy_score(Y_train, train_pred)
            accuracy_scores_test[i][j] = accuracy_score(Y_test, test_pred)
          
    printer = Printerfunctions.Printerfunctions()    
    printer.partD(accuracy_scores_train,accuracy_scores_test,dataset)

# Part e of the project
def part_e(X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot,dataset):
    print("\n""\n")

    print("Starting Project 2: part e")
    print("\n")    
    
    print("Logistic regression using ",dataset," image data")

    print("\nUsing Scikit-Learn:")
    logreg = skl.LogisticRegression(random_state=1,verbose=0,max_iter=1E4,tol=1E-8)
    logreg.fit(X_train,Y_train)
    train_accuracy    = logreg.score(X_train,Y_train)
    test_accuracy     = logreg.score(X_test,Y_test)

    print("Accuracy of training data: ",train_accuracy)
    print("Accuracy of test data: ",test_accuracy)

    epochs = 1000
    mini_batch_size = 12

    scaling=False#True
    printing=False
    
    if (dataset=="MNIST"):
        # 10 categories
        beta = np.random.randn(X_train[0].size,10)
    if (dataset=="WBC"):
        # 2 categories
        beta = np.random.randn(X_train[0].size,2)    
    
    scaling=False
    printing=True
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    accuracy_scores_train = np.zeros((len(eta_vals),len(lmbd_vals)))
    accuracy_scores_test = np.zeros((len(eta_vals),len(lmbd_vals)))    

    print("\nGenerating accuracy scores for Logistic Regression")
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            acc_train,acc_test=functions.LogRegression(X_train,Y_train_onehot,Y_train,X_test,Y_test,epochs,\
                                mini_batch_size,eta,beta,lmbd,scaling,printing)
            accuracy_scores_train[i,j]= acc_train
            accuracy_scores_test[i,j]= acc_test
           
    printer = Printerfunctions.Printerfunctions()
    printer.partE(accuracy_scores_train,accuracy_scores_test,dataset)



if __name__ == '__main__':
    print("---------------------")
    print("Running main function")
    print("---------------------\n")
    warnings.filterwarnings("ignore")
    # The seed
    seed = 2022
    # The degree of the polynomial (number of features) is given by
    n = 6 # change the polynomial level
    # the number of datapoints
    N = 1000    

    X_train, X_test, y_train, y_test = generateDataset(seed,n,N)

    # Note that part_a has a lot of grid searches for optimal parameters and will be very slow
    # to run for large N for the Franke function
    #part_a(X_train,X_test,y_train,y_test,N)
    #part_b_and_c(X_train,X_test,y_train,y_test)
    
    # Generates the dataset for classification
    # using the Wisconsin Breast Cancer dataset
    X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot = generateDatasetWBC()
    dataset="WBC"
    #part_d(X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot,dataset)    
    part_e(X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot,dataset)    
    
    # Generates the dataset for classification
    # using the MNIST dataset
    X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot=generateDatasetImages()
    dataset = "MNIST"
    #part_d(X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot,dataset)
    part_e(X_train,X_test,Y_train,Y_test,Y_train_onehot,Y_test_onehot,dataset)