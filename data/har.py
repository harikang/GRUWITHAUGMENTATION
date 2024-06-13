#for load data
import scipy.io as sio
#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob, os
from numpy import dstack
# Data Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
#manuplate data
import numpy as np
import random
#Encode categorical features (Activity Names) as a one-hot numeric array.
from sklearn.preprocessing import OneHotEncoder
from torch import optim
import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import math
from torch.autograd import Variable
import torch.nn.functional as F
import time

def har():
    #Read Dataset
    path = "./data" #set path
    os.chdir(path) 
    results = pd.DataFrame([])
    list_file = glob.glob("*.csv") #lisiting all the csv file samples
    #print(list_file)

    # function for reading CSV files 
    def reading_file(activity_csv):     
        results = []
        for i in range(len(activity_csv)):
            df = pd.read_csv(activity_csv[i])
            results.append(df.values)    
        return results


    empty_csv = [i for i in list_file if i.startswith('Empty')] #list for empty csv files 
    lying_csv = [i for i in list_file if i.startswith('Lying')] #list for lying csv files 
    sitting_csv = [i for i in list_file if i.startswith('Sitting')] #list for sitting csv files 
    standing_csv = [i for i in list_file if i.startswith('Standing')] #list for satnding csv files 
    walking_csv = [i for i in list_file if i.startswith('Walking')] #list for walking csv files 

    #calling reading_file function  
    empty = reading_file(empty_csv) 
    lying = reading_file(lying_csv)
    sitting = reading_file(sitting_csv)
    standing = reading_file(standing_csv)
    walking = reading_file(walking_csv)


    #function for labeling the samples 
    def label(activity, label):
        list_y = []
        for i in range(len(activity)):
            list_y.append(label)
        return np.array(list_y).reshape(-1, 1) 
        
    walking_label = label(walking, 'walking') 
    empty_label = label(empty, 'empty') 
    lying_label = label(lying, 'lying') 
    sitting_label = label(sitting, 'sitting') 
    standing_label = label(standing, 'standing') 


    #concatenate all the samples into one np array 
    array_tuple = (empty, lying, sitting,standing, walking)
    data_X = np.vstack(array_tuple)

    #concatenate all the label into one array 
    label_tuple = (empty_label, lying_label, sitting_label,standing_label,  walking_label)
    data_y = np.vstack(label_tuple)

    #randomize the sample 
    from sklearn.utils import shuffle
    X, y= shuffle(data_X, data_y)
    return X, y