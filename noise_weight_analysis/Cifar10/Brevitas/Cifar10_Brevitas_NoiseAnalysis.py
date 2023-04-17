# Mark Ashinhust
# Cifar10 Noise Analysis with Cifar10

## MARK: Load Imports

print("\n-------------------------\nLoading Imports\n")

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

print("Done Loading Imports\n-------------------------")
print("\n\nBeginning Cifar10 Neural Networks Analysis with Brevitas\n--------------------------------------------------\n")

## Load The Cifar10 Dataset Using PyTorch
# This will most likely already be downloaded based on the directory but good to check

## Define a PyTorch Device
#
# GPUs can significantly speed-up training of deep neural networks. We check for availability of a GPU and if so define it as target device.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Target device: " + str(device))


print("\nLoading Dataset:\n-------------------------")
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

print("Dataset Values:\n-------------------------")
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

        self.quant_inp = qnn.QuantIdentity(
            bit_width=4, return_quant_tensor=True)

        self.layer1 = qnn.QuantConv2d(
            3, 32, 3, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

        self.relu1 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer2 = qnn.QuantConv2d(
            32, 64, 3, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

        self.relu2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer3 = qnn.QuantLinear(
            64 * 6 * 6, 600, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

        self.relu3 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer4 = qnn.QuantLinear(
            600, 120, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

        self.relu4 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer5 = qnn.QuantLinear(
            120, 10, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.relu1(self.layer1(x))
        x = F.max_pool2d(x, 2)
        x = self.relu2(self.layer2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu3(self.layer3(x))
        x = self.relu4(self.layer4(x))
        x = self.layer5(x)
        return x


# Model setup
model = CIFAR10CNN().to(device)

print("\n\nModel Details:\n-------------------------")

print(model)

## Training and Testing
# 
# The CIFAR10CNN model is a quantized convolutional neural network designed to classify images from the CIFAR-10 dataset. The dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The model architecture consists of the following layers:
# 
# QuantIdentity: This layer quantizes the input image to 4-bit representation.
#
# QuantConv2d (layer1): A 2D quantized convolutional layer with 3 input channels 
# (corresponding to the RGB color channels), 32 output channels, and a 3x3 kernel size. 
# The weights are quantized to 4 bits, and biases are quantized with Int32Bias.
#
# QuantReLU (relu1): A quantized ReLU activation function with a 4-bit output.
#
# Max-pooling: A 2x2 max-pooling operation to reduce the spatial dimensions.
#
# QuantConv2d (layer2): A 2D quantized convolutional layer with 32 input channels, 64 output channels, 
# and a 3x3 kernel size. The weights are quantized to 4 bits, and biases are quantized with Int32Bias.
#
# QuantReLU (relu2): A quantized ReLU activation function with a 4-bit output.
#
# Max-pooling: Another 2x2 max-pooling operation to reduce the spatial dimensions further.
#
# Flatten: A layer that reshapes the tensor into a 1D tensor, preparing it for the fully connected layers.
#
# QuantLinear (layer3): A fully connected (linear) layer with 64 * 6 * 6 input features and 600 output features. 
# The weights are quantized to 4 bits, and biases are quantized with Int32Bias.
#
# QuantReLU (relu3): A quantized ReLU activation function with a 4-bit output.
#
# QuantLinear (layer4): A fully connected (linear) layer with 600 input features and 120 output features. 
# The weights are quantized to 4 bits, and biases are quantized with Int32Bias.
#
# QuantReLU (relu4): A quantized ReLU activation function with a 4-bit output.
#
# QuantLinear (layer5): The final fully connected (linear) layer with 120 input features and 10 output features. The weights are quantized to 4 bits, and biases are quantized with Int32Bias. The 10 output features correspond to the 10 classes in the CIFAR-10 dataset.
# 
# The model processes the input image through these layers in sequence, and the output of the final layer represents the class probabilities.

# Import testing
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import precision_recall_fscore_support

# Initialize the model, optimizer, and criterion
model = CIFAR10CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)

num_epochs = 25
best_test_accuracy = 0
patience = 5
no_improvement_counter = 0

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
    
    # Update the learning rate
    scheduler.step()
    
    # testing phase
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        test_accuracy = 100 * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            
        if no_improvement_counter >= patience:
            print("Early stopping")
            break

        print('Epoch [{}/{}], Test Accuracy: {:.2f}%, Precision: {:.2f}, Recall: {:.2f}, F1 score: {:.2f}'.format(epoch+1, num_epochs, test_accuracy, precision, recall, f1))
