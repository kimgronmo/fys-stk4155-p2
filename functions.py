import numpy as np
from sklearn.metrics import accuracy_score
import math
import autograd.numpy as np
from autograd import grad


# defines some statistical functions

def R2(y_data, y_model):
    return 1 - ( (np.sum((y_data - y_model) ** 2)) / (np.sum((y_data - np.mean(y_data)) ** 2)) )
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# finds the largest score in a R2 array
# will sometimes get overflow nan when doing gd and sgd
def find_largest_score(arr):
    max_val=0.0
    index = -1
    for i in range(len(arr)):
        if (math.isnan(arr[i]))==False:
            #print(arr[i])
            if (arr[i] > max_val) & (arr[i] <= 1.0):
                max_val=arr[i]
                index=i
    return max_val,index



def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]

    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.pinv(D)
    return np.matmul(V,np.matmul(invD,UT))

# defines some basic functions
# given by the projects assignment
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# creates the design matrix, from lecture materials p20
def create_X(x, y, n ):
    if len(x.shape) > 1:
        x = np.ravel(x) 
        y = np.ravel(y)
    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))
    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
    return X
    
# Start by defining the different reggression functions:
# OLS, Ridge, Lasso

# Ordinary Linear Regression
# returns beta values and checks against sklearn for errors
def OLS(xtrain,ytrain):
    # Testing my regression versus sklearn version
    # svd inversion ols regression
    OLSbeta_svd = SVDinv(xtrain.T @ xtrain) @ xtrain.T @ ytrain
    return OLSbeta_svd

# Ridge Regression
# returns beta values  
def RidgeManual(xtrain,lmb,identity,ytrain):
    Ridgebeta = SVDinv((xtrain.T @ xtrain) + lmb*identity) @ xtrain.T @ ytrain
    #print("Ridgebeta in function size is: ",Ridgebeta.size)
    return Ridgebeta
    
# for scaling the learning rate
def learning_schedule(t):
    t0, t1 = 5, 50
    return t0/(t+t1)

def time_decay_rate(t):
    t0, t1 = 1, 10
    return t0/(t+t1)

def CostOLS(y,X,theta):
    return np.sum((y-X @ theta)**2)

def CostRidge(y,X,theta,lmbd):
    return np.sum((y-X @ theta)**2+lmbd*theta.T@theta)        
    

# standard gradient descent
def GD(X_train,y_train,X_test,y_test,epochs,momentum,eta,beta,method,lmbd,printing,scaling):
    n = len(X_train)
    delta  = 1e-8 # Including AdaGrad parameter to avoid possible division by zero
    rho = 0.99 # Value for parameter rho
    # Value for parameters beta1 and beta2, see https://arxiv.org/abs/1412.6980
    beta1 = 0.9
    beta2 = 0.999
    iter=0
    
    ypredict=0 #referenced before assigning
    change=0.0 # for momentum calculations
    if (method=="OLS"):
        if (scaling=="Adagrad"):
            training_gradient = grad(CostOLS,2)
        if (scaling=="RMSprop"):
            training_gradient = grad(CostOLS,2)
        if (scaling=="Adam"):
            training_gradient = grad(CostOLS,2)
    if (method=="Ridge"):
        if (scaling=="Adagrad"):
            training_gradient = grad(CostRidge,2)
        if (scaling=="RMSprop"):
            training_gradient = grad(CostRidge,2)
        if (scaling=="Adam"):
            training_gradient = grad(CostRidge,2)            
    
    for j in range(epochs):
        #print("Starting epoch: ",j)
        Giter=0.0 # for Adagrad
        if (method == "OLS"):
            gradient = (2.0/n)*X_train.T @ ((X_train @ beta)-y_train)
            if (scaling=="None"):
                eta=eta
                new_change = eta*gradient + momentum*change
                beta -= new_change #eta*gradient
                change = new_change                
            if (scaling=="tdr"):
                eta=time_decay_rate(j)
                new_change = eta*gradient + momentum*change
                beta -= new_change #eta*gradient
                change = new_change                    
            if (scaling=="Adagrad"):
                # for Adagrad beta=theta
                gradient = (1.0/n)*training_gradient(y_train, X_train, beta)
                Giter += gradient*gradient
                new_change = gradient*eta/(delta+np.sqrt(Giter))+ momentum*change
                beta -= new_change #eta*gradient
                change = new_change                
            if (scaling=="RMSprop"):
                # for Adagrad beta=theta
                gradient = (1.0/n)*training_gradient(y_train, X_train, beta)
                Giter = (rho*Giter+(1-rho)*gradient*gradient)
                new_change = gradient*eta/(delta+np.sqrt(Giter))+ momentum*change
                beta -= new_change #eta*gradient
                change = new_change   
            if (scaling=="Adam"):
                first_moment = 0.0
                second_moment = 0.0
                iter += 1
                # beta=theta
                gradient = (1.0/n)*training_gradient(y_train, X_train, beta)
                # Computing moments first
                first_moment = beta1*first_moment + (1-beta1)*gradient
                second_moment = beta2*second_moment+(1-beta2)*gradient*gradient
                first_term = first_moment/(1.0-beta1**iter)
                second_term = second_moment/(1.0-beta2**iter)
                # Scaling with rho the new and the previous results
                new_change = eta*first_term/(np.sqrt(second_term)+delta)+ momentum*change
                beta -= new_change #eta*gradient
                change = new_change   
        if (method == "Ridge"):
            lmbd = lmbd
            gradient = (2.0/n)*X_train.T @ ((X_train @ beta)-y_train)+2.0*lmbd*beta
            if (scaling=="None"):
                eta=eta
                new_change = eta*gradient + momentum*change
                beta -= new_change #eta*gradient
                change = new_change                
            if (scaling=="tdr"):
                eta=time_decay_rate(j)
                new_change = eta*gradient + momentum*change
                beta -= new_change #eta*gradient
                change = new_change                       
            if (scaling=="Adagrad"):
                # for Adagrad beta=theta
                gradient = (1.0/n)*training_gradient(y_train, X_train, beta,lmbd)
                Giter += gradient*gradient
                new_change = gradient*eta/(delta+np.sqrt(Giter))+ momentum*change
                beta -= new_change #eta*gradient
                change = new_change                      
            if (scaling=="RMSprop"):
                # for Adagrad beta=theta
                gradient = (1.0/n)*training_gradient(y_train, X_train, beta,lmbd)
                Giter = (rho*Giter+(1-rho)*gradient*gradient)
                new_change = gradient*eta/(delta+np.sqrt(Giter))+ momentum*change
                beta -= new_change #eta*gradient
                change = new_change   
            if (scaling=="Adam"):
                first_moment = 0.0
                second_moment = 0.0
                iter += 1
                # beta=theta
                gradient = (1.0/n)*training_gradient(y_train, X_train, beta,lmbd)
                # Computing moments first
                first_moment = beta1*first_moment + (1-beta1)*gradient
                second_moment = beta2*second_moment+(1-beta2)*gradient*gradient
                first_term = first_moment/(1.0-beta1**iter)
                second_term = second_moment/(1.0-beta2**iter)
                # Scaling with rho the new and the previous results
                new_change = eta*first_term/(np.sqrt(second_term)+delta)+ momentum*change
                beta -= new_change #eta*gradient
                change = new_change                      
                    
                    
                    
                    
        # prints how well we are doing
        ypredict = X_test @ beta
        if (j==(epochs-1)):
            if (printing == False):
                print("\nMethod completed is: ",method," regression")
                print("Number of epochs completed: ", j+1)
                print("My R2 score is: {:.4f}".format(R2(y_test,ypredict)))
                print("Scaling: ",scaling)
    if (printing == True):
        return R2(y_test,ypredict)


# stochastic gradient descent
def SGD(X_train,y_train,X_test,y_test,epochs,momentum,mini_batch_size,eta,beta,lmbd,method,printing,scaling):
    n = len(X_train)
    delta  = 1e-8 # Including AdaGrad parameter to avoid possible division by zero
    rho = 0.99 # Value for parameter rho
    # Value for parameters beta1 and beta2, see https://arxiv.org/abs/1412.6980
    beta1 = 0.9
    beta2 = 0.999
    iter=0
    
    ypredict=0 #referenced before assigning
    change=0.0 # for momentum calculations
    if (method=="OLS"):
        if (scaling=="Adagrad"):
            training_gradient = grad(CostOLS,2)
        if (scaling=="RMSprop"):
            training_gradient = grad(CostOLS,2)
        if (scaling=="Adam"):
            training_gradient = grad(CostOLS,2)
    if (method=="Ridge"):
        if (scaling=="Adagrad"):
            training_gradient = grad(CostRidge,2)
        if (scaling=="RMSprop"):
            training_gradient = grad(CostRidge,2)
        if (scaling=="Adam"):
            training_gradient = grad(CostRidge,2) 
            
    beta=np.copy(beta)
    for j in range(epochs):
        #print("Starting epoch: ",j)
        # Should the batches or training data be shuffled??
        # how to shuffle 2 arrays in the same way??
        
        from sklearn.utils import shuffle 
        X_train,y_train = shuffle(X_train,y_train)

        mini_batches = [X_train[k:k+mini_batch_size]
            for k in range(0,n,mini_batch_size)]

        mini_batches_y = [y_train[k:k+mini_batch_size]
            for k in range(0,n,mini_batch_size)]    
            
        # go through all batches, pick a mini batch and update beta
        m = len(mini_batches)
        counter = 0
        Giter=0.0 # for Adagrad
        for mini_batch in mini_batches:
            # pick a random mini batch
            random_int = np.random.randint(0,len(mini_batches)-1)
            #m = len(mini_batches[random_int])
            if (method == "OLS"):
                gradient = (2.0/mini_batch_size)*mini_batches[random_int].T @ \
                    ((mini_batches[random_int] @ beta)-mini_batches_y[random_int])
                if (scaling=="None"):
                    eta=eta
                    new_change = eta*gradient + momentum*change
                    beta -= new_change #eta*gradient
                    change = new_change                
                if (scaling=="tdr"):
                    eta=time_decay_rate(j*m+counter)
                    counter +=1.0
                    new_change = eta*gradient + momentum*change
                    beta -= new_change #eta*gradient
                    change = new_change                       
                if (scaling=="Adagrad"):
                    # for Adagrad beta=theta
                    gradient = (1.0/n)*training_gradient(mini_batches_y[random_int], mini_batches[random_int], beta)
                    Giter += gradient*gradient
                    new_change = gradient*eta/(delta+np.sqrt(Giter))+ momentum*change
                    beta -= new_change #eta*gradient
                    change = new_change                
                if (scaling=="RMSprop"):
                    # for Adagrad beta=theta
                    gradient = (1.0/n)*training_gradient(mini_batches_y[random_int], mini_batches[random_int], beta)
                    Giter = (rho*Giter+(1-rho)*gradient*gradient)
                    new_change = gradient*eta/(delta+np.sqrt(Giter))+ momentum*change
                    beta -= new_change #eta*gradient
                    change = new_change   
                if (scaling=="Adam"):
                    first_moment = 0.0
                    second_moment = 0.0
                    iter += 1
                    # beta=theta
                    gradient = (1.0/n)*training_gradient(mini_batches_y[random_int], mini_batches[random_int], beta)
                    # Computing moments first
                    first_moment = beta1*first_moment + (1-beta1)*gradient
                    second_moment = beta2*second_moment+(1-beta2)*gradient*gradient
                    first_term = first_moment/(1.0-beta1**iter)
                    second_term = second_moment/(1.0-beta2**iter)
                    # Scaling with rho the new and the previous results
                    new_change = eta*first_term/(np.sqrt(second_term)+delta)+ momentum*change
                    beta -= new_change #eta*gradient
                    change = new_change                       
                    
                    
            if (method == "Ridge"):
                lmbd = lmbd#0.0001 #0.0001
                gradient = (2.0/m)*(mini_batches[random_int].T @ \
                    ((mini_batches[random_int] @ beta)-mini_batches_y[random_int])) \
                        + 2.0*lmbd*beta
                if (scaling=="None"):
                    eta=eta
                    new_change = eta*gradient + momentum*change
                    beta -= new_change #eta*gradient
                    change = new_change                
                if (scaling=="tdr"):
                    eta=time_decay_rate(j*m+counter)
                    counter +=1.0
                    new_change = eta*gradient + momentum*change
                    beta -= new_change #eta*gradient
                    change = new_change                       
                if (scaling=="Adagrad"):
                    # for Adagrad beta=theta
                    gradient = (1.0/n)*training_gradient(mini_batches_y[random_int], mini_batches[random_int], beta,lmbd)
                    Giter += gradient*gradient
                    new_change = gradient*eta/(delta+np.sqrt(Giter))+ momentum*change
                    beta -= new_change #eta*gradient
                    change = new_change                
                if (scaling=="RMSprop"):
                    # for Adagrad beta=theta
                    gradient = (1.0/n)*training_gradient(mini_batches_y[random_int], mini_batches[random_int], beta,lmbd)
                    Giter = (rho*Giter+(1-rho)*gradient*gradient)
                    new_change = gradient*eta/(delta+np.sqrt(Giter))+ momentum*change
                    beta -= new_change #eta*gradient
                    change = new_change   
                if (scaling=="Adam"):
                    first_moment = 0.0
                    second_moment = 0.0
                    iter += 1
                    # beta=theta
                    gradient = (1.0/n)*training_gradient(mini_batches_y[random_int], mini_batches[random_int], beta,lmbd)
                    # Computing moments first
                    first_moment = beta1*first_moment + (1-beta1)*gradient
                    second_moment = beta2*second_moment+(1-beta2)*gradient*gradient
                    first_term = first_moment/(1.0-beta1**iter)
                    second_term = second_moment/(1.0-beta2**iter)
                    # Scaling with rho the new and the previous results
                    new_change = eta*first_term/(np.sqrt(second_term)+delta)+ momentum*change
                    beta -= new_change #eta*gradient
                    change = new_change 

        ypredict = X_test @ beta
        if (j==(epochs-1)):
            if (printing == False):
                print("\nMethod completed is: ",method," regression")
                print("Number of epochs completed: ", j+1)
                print("My R2 score is: {:.4f}".format(R2(y_test,ypredict)))
                print("Scaling: ",scaling)

    if (printing == True):
        return R2(y_test,ypredict)
        
    # to categorical turns our integer vector into a onehot representation
    #    one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    return onehot_vector

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def predict(X):
    #probabilities = self.feed_forward_out(X)
    return np.argmax(X, axis=1)

def to_integer_vector(categorical):
    num = len(categorical)
    category=0
    #print(num)
    int_vector=[]
    for i in range(num):
        if categorical[i][0]==0:
            int_vector.append(1)
        else:
            int_vector.append(0)
    return int_vector
        

def LogRegression(training_data,y_train,Y_train,X_test,y_test,epochs,mini_batch_size,eta,beta,lmb,scaling,printing):     
    n = len(training_data)
    for j in range(epochs):
        #print("Starting epoch: ",j)
        # Should the batches or training data be shuffled??
        mini_batches = [training_data[k:k+mini_batch_size]
            for k in range(0,n,mini_batch_size)]
        mini_batches_y = [y_train[k:k+mini_batch_size]
            for k in range(0,n,mini_batch_size)]    

        # go through all batches, pick a mini batch and update beta
        m = len(mini_batches)
        counter = 0
        #print("m is: ",m," n is: ",n)        
        for mini_batch in mini_batches:
            # pick a random mini batch
            random_int = np.random.randint(0,len(mini_batches)-1)
            lmb = lmb #lambda #0.001
            gradient = (2.0)*(mini_batches[random_int].T @ \
                    (sigmoid(mini_batches[random_int] @ beta) - mini_batches_y[random_int]) \
                        + lmb*beta)
                   
            # scaling the learning rate:
            if (scaling==True):
                eta = learning_schedule(j*m+counter)
                counter += 1
            beta -= eta*gradient
            
        # prints how well we are doing
        ypredict = predict(X_test @ beta)
        ypredict_training = predict(training_data @ beta)
        if (printing==False):
            if (j==(epochs-1)):
                print("\nUsing my own code: ")
                print("Method completed is: Logistic regression")
                print("Number of epochs completed: ", j+1)
                print("Accuracy score on training set: {:.4f}".format(accuracy_score(Y_train, ypredict_training))) 
                print("Accuracy score on test set: {:.4f}".format(accuracy_score(y_test, ypredict)))
    # all data has been calculated return accuracy scores
    return accuracy_score(Y_train, ypredict_training),accuracy_score(y_test, ypredict)