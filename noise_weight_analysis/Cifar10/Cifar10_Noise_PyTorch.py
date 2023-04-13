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
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Model the CIFAR10 Network to the device
model = CIFAR10CNN().to(device)

# Set learning rate for optimizer and criterion
learning_rate=0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Print the model
print(model)



# ## Train and Test
# 
# This code trains a convolutional neural network (CNN) on the Cifar10 dataset using PyTorch. 
# The code uses the torchvision module to download and preprocess the dataset, and defines a CNN with two convolutional layers,
# two max-pooling layers, and three fully connected layers. The code uses the cross-entropy loss function and the 
# Adam optimizer to train the model.
# 
# To run the code, first make sure PyTorch and torchvision are installed on your system.
# Then, run the code in a Python environment. The code should automatically download the Fashion MNIST dataset
#  and train the model for 20 epochs.
# 
# During training, the code prints the loss after every 100 batches. 
# After each epoch, the code evaluates the model on the test set and prints the test accuracy.

num_epochs = 20 
for epoch in range(num_epochs):
    # training phase
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

    # testing phase
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Epoch [{}/{}] -- Test Accuracy --> {:.2f}%'.format(epoch+1, num_epochs, 100 * correct / total))


