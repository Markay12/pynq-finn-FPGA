# Mark Ashinhust
# Cifar10 Noise Analysis with Cifar10

## MARK: Load Imports

from copy import deepcopy
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

# add imports for randomness
import time
import random

import sys

# Brevitas imports
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int32Bias
import torch.nn.functional as F

## Load The Cifar10 Dataset Using PyTorch
# This will most likely already be downloaded based on the directory but good to check

# Load Cifar10 dataset
train_set = torchvision.datasets.CIFAR10(
    "./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.CIFAR10(
    "./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
# Loaders
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=1000)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=1000)
a = next(iter(train_loader))
print(a[0].size())
print(len(train_set))
print("Samples in each set: train = %d, test = %s" %
      (len(train_set), len(train_loader)))
print("Shape of one input sample: " + str(train_set[0][0].shape))


## Data Loader
#
# Using PyTorch dataloader we can create a convenient iterator over the dataset that returns batches of data, rather than requiring manual batch creation.

# set batch size
batch_size = 1000

# Create a DataLoader for a training dataset with a batch size of 100
train_quantized_loader = DataLoader(train_set, batch_size=batch_size)
test_quantized_loader = DataLoader(test_set, batch_size=batch_size)
count = 0

for x, y in train_loader:
    print("Input shape for 1 batch: " + str(x.shape))
    print("Label shape for 1 batch: " + str(y.shape))
    count += 1
    if count == 1:
        break
