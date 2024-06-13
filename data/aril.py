import scipy.io as sio
from torch.utils.data import DataLoader,Dataset
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import os
import tqdm as notebook_tqdm
import math
import torch.nn.functional as F
import time 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 

def aril():
    
    # #fix the random seed
    seed = 1
    deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    #load train data
    data_amp = sio.loadmat('./train_data_split_amp.mat')
    train_data_amp = data_amp['train_data']
    train_data = train_data_amp
    train_label = data_amp['train_activity_label']

    train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.LongTensor)

    #load test data
    data_amp = sio.loadmat('./test_data_split_amp.mat')
    test_data_amp = data_amp['test_data']
    test_data = test_data_amp
    test_label = data_amp['test_activity_label']

    test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)

    train_data = train_data.transpose(1,2)
    test_data = test_data.transpose(1,2)

    X_train_2d = train_data.reshape(train_data.shape[0], -1) #1116x(192*52)
    X_test_2d = test_data.reshape(test_data.shape[0], -1)

    # Standardize the data (mean=0, std=1) using training data
    scaler = StandardScaler().fit(X_train_2d)
    X_train_2d = scaler.transform(X_train_2d)
    # Apply same transformation to test data
    X_test_2d = scaler.transform(X_test_2d)

    X_train = X_train_2d.reshape(train_data.shape) #3116x192x52
    X_test = X_test_2d.reshape(test_data.shape)

    encoder = LabelBinarizer() #labelencoder함수를 가져온다.
    y_train_en=encoder.fit_transform(train_label)
    y_test_en=encoder.fit_transform(test_label)

    return X_train, X_test, y_train_en, y_test_en
