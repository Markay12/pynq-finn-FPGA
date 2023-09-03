"""
Name: Mark Ashinhust
Description: This file serves as the core codebase for conducting experiments involving
             Gaussian Noise application to the Cifar100 CNN model at varying levels of sigma.
             Within this script, the impact of noise on all convolutional layers and
             linear layers at once is tested extensively. The results are visualized through output line
             plots, aiding in the identification of layers that display heightened or diminished
             resilience to noise. These analyses are performed using pre-trained models,
             incorporating a range of weights and activations.

Parameters:  Independent, Gaussian Noise, All Layers
Date: 15 August 2023
"""

## Import Statements

# import from the model functions file
from model_functions import CommonUintActQuant, CommonIntWeightPerTensorQuant, ConvBlock, CIFAR100CNN_4Blocks, CIFAR100CNN_5Blocks
from gauss_functions import gaussian_noise_plots_brevitas, gaussian_noise_plots_brevitas_all, add_gaussian_noise_independent, add_gaussian_noise_proportional, add_gaussian_noise_to_model_brevitas


from copy import deepcopy
import os
from unicodedata import decimal

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
from datetime import datetime

# add imports for randomness
import time
import random

import sys

# Brevitas imports
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int32Bias, Int8WeightPerTensorFloat, Int8ActPerTensorFloat, Int8Bias, Uint8ActPerTensorFloatMaxInit, Int8ActPerTensorFloatMinMaxInit
from brevitas.core.restrict_val import RestrictValueType
import torch.nn.functional as F

# For adaptive learning rate import
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

import importlib
## Imports from utils file for my defined noise functions
import sys

importlib.reload(sys.modules['model_functions'])
importlib.reload(sys.modules['gauss_functions'])


## Extra imports
import numpy as np
import torch
import random
from copy import deepcopy
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import time

# Target device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Target device: " + str(device))

# Define data augmentation transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[.2673342858792401, 0.2564384629170883, 0.27615047132568404])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5088964127604166, 0.48739301317401956, 0.44194221124387256], std=[0.2682515741720801, 0.2573637364478126, 0.2770957707973042])
])

# Apply data augmentation to the training dataset
train_set = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)

# Use the validation transform for the validation dataset
val_set =torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=val_transform)

print("\nTrain and Validation set loaded\n")

# Create the data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)

## Data Loader
# Using PyTorch dataloader we can create a convenient iterator over the dataset that returns batches of data, rather than requiring manual batch creation.

# set batch size
batch_size= 256

# Create CIFAR-100 datasets for training and validation
train_set = torchvision.datasets.CIFAR100(root='./data/', train=True, download=False, transform=train_transform)
val_set =torchvision.datasets.CIFAR100(root='./data/', train=False, download=False, transform=val_transform)

# Create DataLoader objects for training and validation with specified batch size and number of workers
train_quantized_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_quantized_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

# Test and make sure outputs look good
a = next(iter(train_loader))
print(a[0].size())
print(len(train_set))
print("Samples in each set: train = %d, test = %s" % (len(train_set), len(train_loader)))
print("Shape of one input sample: " + str(train_set[0][0].shape))

count = 0

# Show data shapes for one batch
print("\nDataset Shape:\n-------------------------")
for x, y in train_loader:
    print("Input shape for 1 batch: " + str(x.shape))
    print("Label shape for 1 batch: " + str(y.shape))
    count += 1
    if count == 1:
        break

# Load the saved state dictionary from file
# Get model name
model_name = input("Name for the model: ")

state_dict = torch.load(f'..//tested_models//{model_name}.pth', map_location=device)

print("\nState Dict Loaded\n")

# Import testing
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import precision_recall_fscore_support


# Initialize the model, optimizer, and criterion
model = CIFAR100CNN_5Blocks().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0009)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=11, factor=0.1, verbose=True)
criterion = nn.CrossEntropyLoss()

num_epochs = 80
best_test_accuracy = 0
patience = 8
no_improvement_counter = 0

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Test shapes
print(model.conv1.conv.weight.shape)
print(model.conv1.conv.bias)
print(model.conv2.conv.weight.shape)
print(model.fc1.weight.shape)

## Testing Models

# Begin with Mask

print("\nBeginning to test perturbations\n")

perturbations = 20 # Value does not change between tests

## Gaussian Noise Testing

layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'fc1', 'fc2']

## Gaussian Noise
sigma_vector = np.linspace(0, 0.008, 15)

#gaussian_noise_plots_brevitas(perturbations, layer_names, sigma_vector, model, device, val_quantized_loader, 0, model_name)
#gaussian_noise_plots_brevitas(perturbations, layer_names, sigma_vector, model, device, val_quantized_loader, 1, model_name)
#gaussian_noise_plots_brevitas_all(perturbations, layer_names, sigma_vector, model, device, val_quantized_loader, 0, model_name)
gaussian_noise_plots_brevitas_all(perturbations, layer_names, sigma_vector, model, device, val_quantized_loader, 1, model_name)
