

# import functions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import to check code
import sklearn.linear_model as skl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# imports linear regression from scikitlearn to 
# check my code against
from sklearn.metrics import mean_squared_error

# imports Ridge regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# imports functions for real world data
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image
import seaborn as sns

import functions
from sklearn.metrics import accuracy_score
# Where figures and data files are saved..
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)
if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)
if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)
def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)
def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)
def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

class Printerfunctions():
    
    def __init__(self):
        print("\n##### Printing data to files: #####")
        print("Starting printer to generate data:\n")

    def partB1(self,R2_test):
        print("\nPrinting R2 scores for Neural Network Regression to file")    
        sns.set()
        fig, ax = plt.subplots(figsize = (10,10))
        sns.heatmap(R2_test, annot=True, ax=ax, cmap="viridis")
        ax.set_title("R2 scores for Neural Network Regression")
        ax.set_ylabel("$\eta$ from 1e-5 to 1e1")
        ax.set_xlabel("$\lambda$ from 1e-5 to 1e1")
        plt.savefig(os.path.join(os.path.dirname(__file__),'Results/FigureFiles','partB1_NNRegressionR2.png') \
                    ,transparent=True,bbox_inches='tight')

    def partB2(self,MSE_test):
        print("\nPrinting MSE scores for Neural Network Regression to file")    
        sns.set()
        fig, ax = plt.subplots(figsize = (10,10))
        sns.heatmap(MSE_test, annot=True, ax=ax, cmap="viridis")
        ax.set_title("MSE scores for Neural Network Regression")
        ax.set_ylabel("$\eta$ from 1e-5 to 1e1")
        ax.set_xlabel("$\lambda$ from 1e-5 to 1e1")
        plt.savefig(os.path.join(os.path.dirname(__file__),'Results/FigureFiles','partB2_NNRegressionMSE.png') \
                    ,transparent=True,bbox_inches='tight')

    def partD(self,train_accuracy,test_accuracy,dataset):
        print("\nPrinting results for dataset ",dataset," to file")    
        sns.set()
        fig, ax = plt.subplots(figsize = (10,10))
        sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Training Accuracy %s"%dataset)
        ax.set_ylabel("$\eta$ from 1e-5 to 1e1")
        ax.set_xlabel("$\lambda$ from 1e-5 to 1e1")
        plt.savefig(os.path.join(os.path.dirname(__file__),'Results/FigureFiles','training acc part d %s.png'%dataset) \
                    ,transparent=True,bbox_inches='tight')

        fig, ax = plt.subplots(figsize = (10,10))
        sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Test Accuracy %s"%dataset)
        ax.set_ylabel("$\eta$ from 1e-5 to 1e1")
        ax.set_xlabel("$\lambda$ from 1e-5 to 1e1")
        plt.savefig(os.path.join(os.path.dirname(__file__),'Results/FigureFiles','test acc part d %s.png'%dataset) \
                    ,transparent=True,bbox_inches='tight')


    def partE(self,accuracy_scores_train,accuracy_scores_test,dataset):
        print("\nPrinting results for dataset ",dataset," to file")
        sns.set()
        
        fig,ax = plt.subplots(figsize=(10,10))
        sns.heatmap(accuracy_scores_train, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Training Accuracy %s"%dataset)
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.savefig("Results/FigureFiles/accuracy_train_%s.png"%dataset)

        fig,ax = plt.subplots(figsize=(10,10))
        sns.heatmap(accuracy_scores_test, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Test Accuracy %s"%dataset)
        ax.set_ylabel("$\eta$")
        ax.set_xlabel("$\lambda$")
        plt.savefig("Results/FigureFiles/accuracy_test_%s.png"%dataset)