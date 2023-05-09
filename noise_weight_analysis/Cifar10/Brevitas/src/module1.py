
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

# add imports for randomness
import time
import random

import sys

# Brevitas imports
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int32Bias
import torch.nn.functional as F

# For adaptive learning rate import
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split


## Imports from utils file for my defined noise functions
import sys
sys.path.append('C:/Users/ashin/source/repos/Cifar10_Pytorch_NoiseAnalysis/Cifar10_Pytorch_NoiseAnalysis/pynq-finn-FPGA/noise_weight_analysis/utils/')

from noise_functions import mask_noise_plots_brevitas, mask_noise_plots_brevitas_multiple_layers, ber_noise_plot_brevitas, ber_noise_plot_brevitas_multiple_layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Target device: " + str(device))

from torchvision import transforms

# Define data augmentation transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Apply data augmentation to the training dataset
train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)

# Use the validation transform for the validation dataset
val_set =torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=val_transform)

# Create the data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)


a = next(iter(train_loader))
print(a[0].size())
print(len(train_set))

print("Samples in each set: train = %d, test = %s" % (len(train_set), len(train_loader))) 
print("Shape of one input sample: " +  str(train_set[0][0].shape))

## Data Loader
#
# Using PyTorch dataloader we can create a convenient iterator over the dataset that returns batches of data, rather than requiring manual batch creation.

# set batch size
batch_size = 1000

# Create a DataLoader for a training dataset with a batch size of 1000
train_quantized_loader = DataLoader(train_set, batch_size=batch_size)
test_quantized_loader = DataLoader(val_set, batch_size=batch_size)

count = 0

print("\nDataset Shape:\n-------------------------")
for x, y in train_loader:
    print("Input shape for 1 batch: " + str(x.shape))
    print("Label shape for 1 batch: " + str(y.shape))
    count += 1
    if count == 1:
        break


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=4, return_quant_tensor=True)

        self.layer1 = qnn.QuantConv2d(3, 32, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu1 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer2 = qnn.QuantConv2d(32, 32, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer3 = qnn.QuantConv2d(32, 64, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu3 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer4 = qnn.QuantConv2d(64, 64, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu4 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer5 = qnn.QuantConv2d(64, 64, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu5 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.fc1 = qnn.QuantLinear(64 * 8 * 8, 512, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu6 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.fc2 = qnn.QuantLinear(512, 10, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = F.max_pool2d(x, 2)

        x = self.relu3(self.layer3(x))
        x = self.relu4(self.layer4(x))
        x = F.max_pool2d(x, 2)

        x = self.relu5(self.layer5(x))

        x = x.view(x.size(0), -1)

        x = self.relu6(self.fc1(x))
        x = self.fc2(x)

        return x


# Import testing
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import precision_recall_fscore_support

# Initialize the model, optimizer, and criterion
model = CIFAR10CNN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
criterion = nn.CrossEntropyLoss()

num_epochs = 80
best_test_accuracy = 0
patience = 8
no_improvement_counter = 0


# Create an instance of your neural network model
model = CIFAR10CNN().to(device)

model_name = input("Name of the Model to Test: ")

# Load the saved state dictionary from file
state_dict = torch.load(model_name)

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Test shapes
print(model.layer1.weight.shape)
print(model.layer2.weight.shape)
print(model.fc1.weight.shape)

## Testing Models

# Begin with Mask

random.seed(42)

perturbations = 15 # Value does not change between tests

layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
layer_combinations = [['layer1', 'layer2', 'layer3', 'layer4', 'layer5']]


p_values = [1, 0.5, 0.25]
gamma_values = np.linspace(0.001, 0.1, 5)
sigma = np.linspace(0.0, 0.2, 10)

# Test independently with Independent and Proportional
mask_noise_plots_brevitas(perturbations, layer_names, p_values, gamma_values, model, device, sigma, 1)
mask_noise_plots_brevitas(perturbations, layer_names, p_values, gamma_values, model, device, sigma, 0)

# Test all layers together with independent and proportional
mask_noise_plots_brevitas_multiple_layers(perturbations, layer_combinations, p_values, gamma_values, model, device, sigma, 1)
mask_noise_plots_brevitas_multiple_layers(perturbations, layer_combinations, p_values, gamma_values, model, device, sigma, 0)


## BER Noise Testing

layer_names.append('fc1', 'fc2')
layer_combinations = [['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'fc1', 'fc2']]

ber_vals = np.linspace(1e-5, 0.01, 15)

ber_noise_plot_brevitas(perturbations, layer_names, ber_vals, model, device, test_quantized_loader)
ber_noise_plot_brevitas_multiple_layers(perturbations, layer_combinations, ber_vals, model, device, test_quantized_loader)

## Gaussian Noise Testing

## Gaussian Noise
sigma_vector = np.linspace(0, 0.05, 15)

gaussian_noise_plots_brevitas(perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, 0)
gaussian_noise_plots_brevitas(perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, 1)