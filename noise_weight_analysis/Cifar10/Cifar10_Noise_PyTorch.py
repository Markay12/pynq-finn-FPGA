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

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

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

# Define the train and validation set sizes
train_size = int(len(train_set) * 0.8)
val_size = len(train_set) - train_size

# Split the dataset into train and validation sets
train_data, val_data = random_split(train_set, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1000)

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


class CIFAR10CNN(nn.Module):

    def __init__(self):
        super(CIFAR10CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=128*4*4, out_features=1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)
        out = self.fc3(out)

        return out



model = CIFAR10CNN().to(device)


# Create an adaptive learning rate 
learning_rate=0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For the adaptive learning rate
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
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


# updating training to add adaptive lr

num_epochs = 20
for epoch in range(num_epochs):
    # training phase
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # validation phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    val_accuracy = 100 * correct_val / total_val

    # call scheduler to adjust learning rate
    scheduler.step(val_loss)

    # print loss and accuracy
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'
          .format(epoch+1, num_epochs, train_loss/len(train_loader), train_accuracy, val_loss/len(val_loader), val_accuracy))




