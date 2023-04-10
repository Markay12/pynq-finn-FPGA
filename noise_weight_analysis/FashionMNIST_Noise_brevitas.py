#!/usr/bin/env python
# coding: utf-8

# # Test Different Weight and Bias Values on Fashion MNIST Dataset using Brevitas

# Load imports. Make sure that onnx is imported before Torch. __ONNX is always imported before Torch__

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

# Brevitas imports
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int32Bias
import torch.nn.functional as F

# ## Load The Fashion MNIST Dataset Using PyTorch
#
# Here we load the Fashion MNIST dataset using `torchvision.datasets.FashionMNIST`, and specify that we want to download the dataset if it isn't already downloaded. We also use the transform argument to apply the `transforms.Compose([transforms.ToTensor()])` transformation to each image, which converts the images from PIL Images to PyTorch Tensors.
#
# We then wrap the `train_set` and `test_set` datasets in PyTorch `DataLoader` objects, which allow us to load the data in batches for more efficient processing during training and testing.
#
# Finally, we use `next(iter(train_loader))` to retrieve the first batch of training data from the `train_loader`, and print the size of the batch and the total number of training examples in train_set.


# Load Fashion MNIST dataset
train_set = torchvision.datasets.FashionMNIST(
    "./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST(
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


# ## Data Loader
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


# # Define a PyTorch Device
#
# GPUs can significantly speed-up training of deep neural networks. We check for availability of a GPU and if so define it as target device.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Target device: " + str(device))


# ## Define the Quantized Model

# Update module to use Brevitas

########################################################
# Layer by Layer Breakdown
# -----------------------------
# layer1: Convolutional Layer with 1 input channel, 32 output channels and a 3x3 kernel size
#         Weights in this channel are quantized to 4 bits with a quantized input of 8 bits.
#         Output passed through a ReLU activation then batch normalized and lastly max-pooled with 2x2 kernel.
#
# layer 2: Convolutional layer with 32 input channels, 64 output channels and 3x3 kernel.
#          Weights are quantized to 4 bits and the input is quantized to 8 bits.
#          Passed through an ReLU activation, batch normalized, the max-pooled as well with a 2x2 kernel size.
#
# fc1:     Fully connected layer with 64*6*6 input features and 600 output features.
#          The weights are quantized to 4 bits and the output is quantized to 8 bits.
#
# fc2:     Fully connected layer with 600 input features and 120 output features.
#          Quantized weights to 4 bits and input quantized to 8 bits
#
# fc3:     This is the final fully connected layer with 120 input features and 10 output features.
#          The 10 output corresponds to the FashionMNIST classes.
#          Weights are again quantized to 4 bits and input to 8 bits.
#

# setup class for the neural network building
class FashionCNN(nn.Module):

    def __init__(self):
        super(FashionCNN, self).__init__()

        self.quant_inp = qnn.QuantIdentity(
            bit_width=4, return_quant_tensor=True)

        # One input channel for the 28x28 images in grayscale
        # in_channels = 1, out_channels = 6, kernel_size = 5
        self.layer1 = qnn.QuantConv2d(
            1, 6, 5, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

        self.relu1 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        # input channels is the output of the last Conv2d
        # in_channels = 6, out_channels = 16, kernel_size = 5
        self.layer2 = qnn.QuantConv2d(
            6, 16, 5, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

        self.relu2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        # Fully Connected Layers using Brevitas
        self.layer3 = qnn.QuantLinear(
            16 * 4 * 4, 120, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

        self.relu3 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer4 = qnn.QuantLinear(
            120, 84, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

        self.relu4 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer5 = qnn.QuantLinear(
            84, 10, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

    # feed forward
    def forward(self, x):

        # forward pass
        x = self.quant_inp(x)
        x = self.relu1(self.layer1(x))
        x = F.max_pool2d(x, 2)
        x = self.relu2(self.layer2(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(x.shape[0], -1)
        x = self.relu3(self.layer3(x))
        x = self.relu4(self.layer4(x))
        x = self.layer5(x)

        return x      # output


model = FashionCNN().to(device)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)


# ## Train and Test
#
# This code trains a convolutional neural network (CNN) on the Fashion MNIST dataset using PyTorch. The code uses the torchvision module to download and preprocess the dataset, and defines a CNN with two convolutional layers, two max-pooling layers, and three fully connected layers. The code uses the cross-entropy loss function and the Adam optimizer to train the model.
#
# To run the code, first make sure PyTorch and torchvision are installed on your system. Then, run the code in a Python environment. The code should automatically download the Fashion MNIST dataset and train the model for 20 epochs.
#
# During training, the code prints the loss after every 100 batches. After each epoch, the code evaluates the model on the test set and prints the test accuracy.


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

        print('Epoch [{}/{}], Test Accuracy: {:.2f}%'.format(epoch +
              1, num_epochs, 100 * correct / total))


# ## Changing Model Parameters (Weights/Biases)
#
# First we show how to get the trained_state_dictionary which shows all weights, biases etc. Then we also show how to extract just one layers weights.
#
# Then we run a script that defines a function called test that evaluates the performance of a given model on a given dataset. Then it defines two more functions called add_noise and add_noise_to_model, which add Gaussian noise to the weight and bias parameters of the given model, respectively. Finally, the script applies the add_noise_to_model function to the given model for a single layer, specified in the layers list, with increasing standard deviation values in sigma_vector, and measures the resulting test accuracy using the test function. The output is a series of test accuracy values for the model with increasing noise added to the specified layer.


# Get all model Parameters
trained_state_dict = model.state_dict()


def test(model, test_loader):
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

    accuracy = 100 * correct / total
    return accuracy

# Sigma is weight dependent
# Range is clipped to wmin/wmax

# Add noise to weight and or bias matrices.
# The weight is a centered normal around the weight values with a sigma difference.


# Ensure Randomness
random.seed(time.time())


def add_noise(matrix, sigma):

    noised_weight = np.random.normal(loc=matrix, scale=sigma)

    return noised_weight


def add_noise_to_model(model, layer_names, sigma, num_perturbations):
    modified_models = []

    for _ in range(num_perturbations):
        modified_model = deepcopy(model)

        # add noise to the modified model
        for layer_name in layer_names:
            layer = getattr(modified_model, layer_name)

            with torch.no_grad():
                # Get the weight and bias tensors
                weight = layer.weight.detach().numpy()
                bias = layer.bias.detach().numpy() if layer.bias is not None else None

                # Add noise to the weight and bias tensors
                noised_weight = add_noise(weight, sigma)
                if bias is not None:
                    noised_bias = add_noise(bias, sigma)

                # Update the layer's weight and bias tensors with the noised values
                layer.weight = torch.nn.Parameter(
                    torch.tensor(noised_weight, dtype=torch.float))
                if bias is not None:
                    layer.bias = torch.nn.Parameter(
                        torch.tensor(noised_bias, dtype=torch.float))

        modified_models.append(modified_model)

    return modified_models


# Define the standard deviation values to test
sigma_vector = np.linspace(0, 0.2, 11)

# Define the layer to add noise to all layers
# num_layers = 13
# layers = range(1, num_layers + 1)

# add noise to only the first layer
layers = ['layer1']

# test 5 times and average to smooth out the curve
num_perturbations = 10

print("\nTesting Perturbations for Layer 1:\n")

# Loop over each standard deviation value in sigma_vector
for s in range(len(sigma_vector)):

    # Add noise to the model for the defined layer only
    noisy_models = add_noise_to_model(
        model, layers, sigma_vector[s], num_perturbations)

    accuracies = []
    # Test the accuracy of each noisy model and append the result to the accuracies list
    for noisy_model in noisy_models:
        # Move the model back to the target device
        noisy_model.to(device)

        accuracies.append(test(noisy_model, test_quantized_loader))

    # Calculate the average accuracy and print the result
    avg_accuracy = sum(accuracies) / len(accuracies)
    print("Sigma Value:", sigma_vector[s],
          ", Average Accuracy:", avg_accuracy, "%")


# Define layer names
layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']

if not os.path.exists("noise_plots_brevitas"):
    os.makedirs("noise_plots_brevitas")

plt.style.use('default')

# Initialize the standard deviation values
sigma_vector = np.linspace(0, 0.2, 31)

# Loop over each layer and plot the test accuracy as a function of the standard deviation for that layer
for layer in layer_names:

    # Initialize a list to store the test accuracies for this layer
    test_accs = []

    # Iterate over the standard deviation values and add noise to the model for this layer only
    for sigma in sigma_vector:

        # Add noise to the model for the defined layer only
        noisy_models = add_noise_to_model(
            model, [layer], sigma, num_perturbations)

        accuracies = []
        # Test the accuracy of each noisy model and append the result to the accuracies list
        for noisy_model in noisy_models:
            # Move the model back to the target device
            noisy_model.to(device)

            accuracies.append(test(noisy_model, test_quantized_loader))

        # Calculate the average accuracy and print the result
        avg_accuracy = sum(accuracies) / len(accuracies)

        # Append the average accuracy to the test_accs list
        test_accs.append(avg_accuracy)

        print("Sigma Value: {}, Average Accuracy: {}%".format(sigma, avg_accuracy))

    # Plot the test accuracies as a function of the standard deviation for this layer
    plt.plot(sigma_vector, test_accs)
    plt.xlabel('Standard Deviation')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of Noise on Test Accuracy ({})'.format(layer))
    plt.savefig(
        "noise_plots_brevitas/{}.png".format(layer))
    plt.show()

    print('Done with Plot {}'.format(layer))
