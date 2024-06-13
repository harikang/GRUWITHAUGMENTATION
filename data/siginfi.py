import scipy.io as sio
from torch.utils.data import DataLoader,Dataset
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import tqdm as notebook_tqdm
import math
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit

def signfi():
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
        
    csi_lab =sio.loadmat('dataset_lab_276_dl.mat')
    csi_home =sio.loadmat('dataset_home_276.mat')

    data_lab = csi_lab['csid_lab']
    label_lab = csi_lab['label_lab']

    data_home = csi_home['csid_home']
    label_home = csi_home['label_home']

    csi_abs_home =torch.FloatTensor(np.abs(data_home)) #amp값
    csi_abs_lab =torch.FloatTensor(np.abs(data_lab)) #amp값
    csi_abs = csi_abs_home

    data = csi_abs.permute(3,0,1,2)

    input_2d = data.reshape(data.shape[0], -1)

    # Standardize the data (mean=0, std=1) using training data
    scaler = StandardScaler().fit(input_2d)
    input_2d = scaler.transform(input_2d)


    input = input_2d.reshape(data.shape) #3116x192x52
    input = torch.tensor(input)

    input.shape
    data = input.reshape(input.shape[0],input.shape[1],-1)

    encoder = OneHotEncoder(sparse=False)
    label_home = encoder.fit_transform(label_home)
    label_lab = encoder.fit_transform(label_lab)
    label_plus = torch.cat([torch.FloatTensor(label_home),torch.FloatTensor(label_lab)],0)
    return data, label_home, label_lab, label_plus