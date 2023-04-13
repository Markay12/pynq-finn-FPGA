## Cifar 10 Neural Network Perturbation Analysis using PyTorch

# Mark Ashinhust
# April 12, 2023

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torchvision import datasets, transforms

# imports for randomness and setting seed
import time
import random

import os

# setting random seed for random variable noise each time
random.seed(time.time())

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Target device: " + str(device))

## Load and Normalize Cifar10 Dataset
# The output of the torchvision dataest are PILImage images of range [0, 1]. These are then transformed into Tensors of normalized range being [-1, 1].

# Load CIFAR-10 dataset
train_set = torchvision.datasets.CIFAR10("./data", download=True, transform=
                                         transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.CIFAR10("./data", download=True, train=False, transform=
                                        transforms.Compose([transforms.ToTensor()]))

# Loaders
train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=1000)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=1000)

a = next(iter(train_loader))
print(a[0].size())
print(len(train_set))

print("Samples in each set: train = %d, test = %s" % (len(train_set), len(train_loader))) 
print("Shape of one input sample: " +  str(train_set[0][0].shape))

# set batch size
batch_size = 1000

# Create a DataLoader for a training dataset with a batch size of 100
train_quantized_loader = DataLoader(train_set, batch_size=batch_size)
test_quantized_loader = DataLoader(test_set, batch_size=batch_size)

# Initialize a variable to count the number of batches in the train_loader
count = 0

# Iterate through the train_loader
for x,y in train_loader:

  # Print the shape of the input and label tensors for the current batch
  print("Input shape for 1 batch: " + str(x.shape))
  print("Label shape for 1 batch: " + str(y.shape))

  # Increment the count variable
  count += 1

  # Exit the loop after printing the first batch
  if count == 1:
    break

# Determine if we can use the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Target device: " + str(device))

# Update the model architecture for CIFAR-10
class CIFAR10CNN(nn.Module):
    
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out


# Model the CIFAR10 Network to the device
model = CIFAR10CNN().to(device)

# Set learning rate for optimizer and criterion
learning_rate=0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Print the model
print(model)



