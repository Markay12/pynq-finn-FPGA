"""
Name: Mark Ashinhust
Description: This file serves as the core codebase for conducting experiments involving
             Gaussian Noise application to the Cifar100 CNN model at varying levels of sigma.
             Within this script, the impact of noise on all convolutional layers and
             linear layers at once is tested extensively. The results are visualized through output line
             plots, aiding in the identification of layers that display heightened or diminished
             resilience to noise. These analyses are performed using pre-trained models,
             incorporating a range of weights and activations.

Parameters:  Proportional, Gaussian Noise, All Layers
Date: 15 August 2023
"""

## Import Statements

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
from noise_functions import mask_noise_plots_brevitas, mask_noise_plots_brevitas_multiple_layers, ber_noise_plot_brevitas, ber_noise_plot_brevitas_multiple_layers, gaussian_noise_plots_brevitas, gaussian_noise_plots_brevitas_all

importlib.reload(sys.modules['noise_functions'])

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

state_dict = torch.load(f'//home//mashinhu//Desktop//Cifar100Testing//tested_models//{model_name}.pth', map_location=device)

print("\nState Dict Loaded\n")

# Import testing
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import precision_recall_fscore_support

class CommonUintActQuant(Uint8ActPerTensorFloatMaxInit):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    max_val = 6.0
    restrict_scaling_type = RestrictValueType.LOG_FP


class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
    """
    Common per-tensor weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            weight_bit_width,
            act_bit_width=16,
            stride=1,
            padding=0,
            groups=1,
            bn_eps=1e-5,
            activation_scaling_per_channel=False,
            normalization_func='batch',
            dropout=0.0,
            return_quant_tensor=True ):
        super(ConvBlock, self).__init__()
        self.conv = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            weight_quant=CommonIntWeightPerTensorQuant,
            weight_bit_width=weight_bit_width)

        if normalization_func == 'batch':
            self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        elif normalization_func == 'instance':
            self.bn = nn.InstanceNorm2d(num_features=out_channels, eps=bn_eps)
        elif normalization_func == 'none':
            self.bn = lambda x: x
        else:
            raise NotImplementedError(f'{normalization_func} is not a supported normalization function')
        self.dropout = nn.Dropout2d(p=dropout)
        self.activation = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            scaling_per_channel=False,
            return_quant_tensor=return_quant_tensor
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x


class CIFAR100CNN_4Blocks(nn.Module):
    def __init__(self,layer_one_bw=16,layer_two_bw=16,layer_three_bw=16,layer_four_bw=16,fcn_bw=16,act_bits=16,num_classes=100,normalization_func='batch',dropout=0.1):
        super(CIFAR100CNN_4Blocks, self).__init__()

        # Block 1
        self.conv1 = ConvBlock(3,64,3,weight_bit_width=layer_one_bw, act_bit_width=act_bits,stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv2 = ConvBlock(64,64,3,weight_bit_width=layer_one_bw, act_bit_width=act_bits,stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)

        # Block 2
        self.conv3 = ConvBlock(64,128,3,weight_bit_width=layer_two_bw, act_bit_width=act_bits,stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv4 = ConvBlock(128,128,3,weight_bit_width=layer_two_bw, act_bit_width=act_bits,stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)


        # Block 3
        self.conv5 = ConvBlock(128,128,3,weight_bit_width=layer_three_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv6 = ConvBlock(128,128,3,weight_bit_width=layer_three_bw, act_bit_width=act_bits,stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)


        # Block 4
        self.conv7 = ConvBlock(128,256,3,weight_bit_width=layer_four_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv8 = ConvBlock(256,256,3,weight_bit_width=layer_four_bw,act_bit_width=act_bits, stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)

        # Flatten then FCN
        self.fc1 = qnn.QuantLinear(256, 256, weight_bit_width=fcn_bw,bias=False)
        self.relu1 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bits,
            scaling_per_channel=False,
            return_quant_tensor=True
            )

        self.fc2 = qnn.QuantLinear(256, 100, weight_bit_width=fcn_bw,bias=False)

    def forward(self, x):
        # Unit 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        # Unit 2
        x = self.conv3(x)
        x = self.conv4(x)
        #x = F.max_pool2d(x, 2)

        # Unit 3
        x = self.conv5(x)
        x = self.conv6(x)
        #x = F.max_pool2d(x, 2)

        # Unit 4
        x = self.conv7(x)
        x = self.conv8(x)
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x


class CIFAR100CNN_5Blocks(nn.Module):
    def __init__(self,layer_one_bw=16,layer_two_bw=16,layer_three_bw=16,layer_four_bw=16,layer_five_bw=16,fcn_bw=16,act_bits=16,num_classes=100,normalization_func='batch',dropout=0.1):
        super(CIFAR100CNN_5Blocks, self).__init__()

        # Block 1
        self.conv1 = ConvBlock(3,64,3,weight_bit_width=layer_one_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv2 = ConvBlock(64,64,3,weight_bit_width=layer_one_bw, act_bit_width=act_bits,stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)

        # Block 2
        self.conv3 = ConvBlock(64,128,3,weight_bit_width=layer_two_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv4 = ConvBlock(128,128,3,weight_bit_width=layer_two_bw,act_bit_width=act_bits, stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)


        # Block 3
        self.conv5 = ConvBlock(128,128,3,weight_bit_width=layer_three_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv6 = ConvBlock(128,128,3,weight_bit_width=layer_three_bw,act_bit_width=act_bits, stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)


        # Block 4
        self.conv7 = ConvBlock(128,256,3,weight_bit_width=layer_four_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv8 = ConvBlock(256,256,3,weight_bit_width=layer_four_bw,act_bit_width=act_bits, stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)

        # Block 5
        self.conv9 = ConvBlock(256,256,3,weight_bit_width=layer_five_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv10 = ConvBlock(256,256,3,weight_bit_width=layer_five_bw,act_bit_width=act_bits, stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)

        # Flatten then FCN
        self.fc1 = qnn.QuantLinear(256, 256, weight_bit_width=fcn_bw,bias=False)
        self.relu1 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bits,
            scaling_per_channel=False,
            return_quant_tensor=True
            )

        self.fc2 = qnn.QuantLinear(256, 100, weight_bit_width=fcn_bw,bias=False)

    def forward(self, x):
        # Organize Image
        # Unit 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        # Unit 2
        x = self.conv3(x)
        x = self.conv4(x)
        #x = F.max_pool2d(x, 2)

        # Unit 3
        x = self.conv5(x)
        x = self.conv6(x)
        #x = F.max_pool2d(x, 2)

        # Unit 4
        x = self.conv7(x)
        x = self.conv8(x)
        x = F.max_pool2d(x, 2)

        # Unit 5
        x = self.conv9(x)
        x = self.conv10(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x

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


"""
The function sets the model to evaluation mode, disables gradient computation,
and iterates through the test data loader, passing each batch of images through the
model to get predicted labels. It then compares the predicted labels to the true labels
and calculates the percentage of correct predictions (accuracy) over the entire test
dataset. Finally, the function returns the accuracy value as a float.
"""

def test(model, val_loader, device):

    # testing phase
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    accuracy = 100 * correct / total
    return accuracy

#################################################
## Gaussian Noise Section
###############################################

"""
This function adds Gaussian noise to a given matrix. The sigma parameter controls
the standard deviation of the noise distribution. The function returns the original
matrix with added noise.
"""

def add_gaussian_noise_independent(matrix, sigma):
    noised_matrix = matrix + np.random.normal(0, scale=sigma)
    return noised_matrix

def add_gaussian_noise_proportional(matrix, sigma):
    noised_matrix = matrix * np.random.normal(1, scale=sigma)
    return noised_matrix


def return_noisy_matrix(independent, dim_matrix, sigma, device):

    if (independent):
        noised_matrix = torch.randn(dim_matrix) * sigma
    else:
        noised_matrix = torch.randn(dim_matrix) * sigma + 1

    return noised_matrix


"""
The add_gaussian_noise_to_model_brevitas function takes a Brevitas model, a list of layer names to which noise should be added,
a standard deviation sigma of the Gaussian noise, and an integer num_perturbations representing the number of times to apply
noise to the specified layers.
For each perturbation, the function makes a deep copy of the original model and adds Gaussian noise to the specified layers
weights and biases using the add_noise function. The noisy weights and biases are then used to update the corresponding layer's
weight and bias tensors.
After each perturbation, the modified model is added to a list, which is then returned once all perturbations have been applied.
"""

def add_gaussian_noise_to_model_brevitas(model, layer_names, sigma, num_perturbations, analog_noise_type):


    modified_models = []

    for _ in range(num_perturbations):

        modified_model = deepcopy(model)

        # add noise to the modified model
        for layer_name in layer_names:

            random.seed(datetime.now().timestamp())

            layer = getattr(modified_model, layer_name)

            with torch.no_grad():

                if layer_name.lower().startswith("c"):
                    weight = layer.conv.weight.cpu().clone().detach().numpy()
                    bias = layer.conv.bias.cpu().detach().numpy() if layer.conv.bias is not None else None
                else:
                    weight = layer.weight.cpu().clone().detach().numpy()
                    bias = layer.bias.cpu().detach().numpy() if layer.bias is not None else None

                # Add noise to the weight and bias tensors
                if (analog_noise_type):
                    noised_weight = add_gaussian_noise_independent(weight, sigma)

                    if bias is not None:
                        noised_bias = add_gaussian_noise_independent(bias, sigma)
                else:
                    noised_weight = add_gaussian_noise_proportional(weight, sigma)

                    if bias is not None:
                        noised_bias = add_gaussian_noise_proportional(bias, sigma)


                # Update layer weights and bias tensors with noised values
                if layer_name.lower().startswith("c"):
                    layer.conv.weight = torch.nn.Parameter(torch.tensor(noised_weight, dtype = torch.float))

                    if bias is not None:
                        layer.conv.bias = torch.nn.Parameter(torch.tensor(noised_bias, dtype = torch.float))

                else:
                    layer.weight = torch.nn.Parameter(torch.tensor(noised_weight, dtype = torch.float))

                    if bias is not None:
                        layer.bias = torch.nn.Parameter(torch.tensor(noised_bias, dtype = torch.float))

        modified_models.append(modified_model)

    return modified_models


"""
This function takes in a PyTorch model and modifies the weights and biases of specified layers by adding
Gaussian noise with a given standard deviation. The function creates num_perturbations modified models,
each with the same architecture as the original model but with different randomly perturbed weights and biases in the specified layers.
The layers to be perturbed are specified by a list of layers, where each layer is identified by its index in the model's layers.
The function returns a list of the modified models.
"""

def add_gaussian_noise_to_model_pytorch(model, layers, sigma, num_perturbations):

    # keep all modified models accuracy so we can get a correct average
    modified_models = []

    for _ in range(num_perturbations):

        # create individual modified model
        modified_model = deepcopy(model)

        for i in range(len(layers)):
            layer_name = f'layer{i + 1}'
            layer = getattr(modified_model, layer_name)

            # get weight and bias for layer
            weight_temp = layer[0].weight.data.cpu().detach().numpy()
            bias_temp = layer[0].bias.data.cpu().detach().numpy()


            # add noise to the weight and bias
            noise_w = add_noise(weight_temp, sigma)
            noise_b = add_noise(bias_temp, sigma)

            # place values back into the modified model for analysis
            layer[0].weight.data = torch.from_numpy(noise_w).float()
            layer[0].bias.data = torch.from_numpy(noise_b).float()

        # append this modified model to the list
        modified_models.append(modified_model)


    return modified_models


"""
This function gaussian_noise_plots_brevitas adds Gaussian noise to a given model for each layer and creates plots to visualize the effect of noise on test accuracy.
The function takes in the following parameters:
    num_perturbations: The number of times to perturb the model for each value of standard deviation.
    layer_names: A list of layer names to add noise to.
    sigma_vector: A list of standard deviation values to add noise with.
    model: The model to perturb.
    device: The device to use for computation.
The function first creates a folder to store the noise plots, if it does not already exist. Then, for each layer in layer_names,
it loops over each value of standard deviation in sigma_vector and adds Gaussian noise to the model for the current layer only.
It then tests the accuracy of each noisy model and appends the result to a list of test accuracies.
After testing for all standard deviation values, the function plots the test accuracy as a function of the standard deviation
for the current layer and saves the plot to a file. It then computes the average test accuracy across all layers for each
standard deviation value and plots the averaged test accuracies as a function of the standard deviation.
Finally, the function saves the average test accuracy plot to a file and displays it.
"""

def gaussian_noise_plots_brevitas(num_perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, analog_noise_type, model_name):

    if not os.path.exists(f"noise_plots_brevitas/gaussian_noise/{model_name}"):
        os.makedirs(f"noise_plots_brevitas/gaussian_noise/{model_name}")

    plt.style.use('default')

    # Create a list to store the test accuracies for all layers
    all_test_accs = []

    if (analog_noise_type):
        csv_file_path = os.path.join(f"noise_plots_brevitas/gaussian_noise/{model_name}", "raw_data_independent.csv")
    else:
        csv_file_path = os.path.join(f"noise_plots_brevitas/gaussian_noise/{model_name}", "raw_data_proportional.csv")



    # Create a CSV file to store the raw data
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Layer", "Sigma Value", "Average Accuracy"])

        # Loop over each layer and plot the test accuracy as a function of the standard deviation for that layer
        for layer in layer_names:

            ## Random seed
            random.seed(datetime.now().timestamp())

            print("Random Seed Generated")

            # Initialize a list to store the test accuracies for this layer
            test_accs = []

            # Iterate over the standard deviation values and add noise to the model for this layer only
            for sigma in sigma_vector:
                # Add noise to the model for the defined layer only
                noisy_models = add_gaussian_noise_to_model_brevitas(
                    model, [layer], sigma, num_perturbations, analog_noise_type)
                accuracies = []

                # Test the accuracy of each noisy model and append the result to the accuracies list
                for noisy_model in noisy_models:
                    # Move the model back to the target device
                    noisy_model.to(device)
                    accuracies.append(test(noisy_model, test_quantized_loader, device))

                # Calculate the average accuracy and print the result
                avg_accuracy = sum(accuracies) / len(accuracies)
                # Append the average accuracy to the test_accs list
                test_accs.append(avg_accuracy)

                # Write the raw data to the CSV file
                writer.writerow([layer, sigma, avg_accuracy])

                if analog_noise_type:
                    print("Independent: Layer: {}, Sigma Value: {}, Average Accuracy: {}%".format(
                        layer, sigma, avg_accuracy))
                else:
                    print("Proportional: Layer: {}, Sigma Value: {}, Average Accuracy: {}%".format(
                        layer, sigma, avg_accuracy))

            # Store the test accuracies for this layer in the all_test_accs list
            all_test_accs.append(test_accs)
            # Plot the test accuracies as a function of the standard deviation for this layer
            plt.plot(sigma_vector, test_accs,
                     label='{} Accuracy'.format(layer))

        # Compute the average test accuracy across all layers for each standard deviation value
        avg_test_accs = [sum(x) / len(x) for x in zip(*all_test_accs)]

        # Plot the averaged test accuracies as a function of the standard deviation
        plt.plot(sigma_vector, avg_test_accs, label='Average',
                 linewidth=3, linestyle='--', color="black")

        plt.xlabel('Standard Deviation')
        plt.ylabel('Test Accuracy')
        plt.title('Effect of Noise on Test Accuracy (Individual Layers and Average)')
        plt.legend()

        if analog_noise_type:
            plt.savefig(f"noise_plots_brevitas/gaussian_noise/{model_name}/individual_and_average_independent.png")
        else:
            plt.savefig(f"noise_plots_brevitas/gaussian_noise/{model_name}/individual_and_average_proportional.png")

        plt.show()
        plt.clf()


def gaussian_noise_plots_brevitas_all(num_perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, analog_noise_type, model_name):

    if not os.path.exists(f"noise_plots_brevitas/gaussian_noise/{model_name}"):
        os.makedirs(f"noise_plots_brevitas/gaussian_noise/{model_name}")

    plt.style.use('default')

    # Create a CSV file to store the raw data
    if (analog_noise_type):
        csv_file_path = os.path.join(f"noise_plots_brevitas/gaussian_noise/{model_name}", "raw_data_independent_all_layers.csv")
    else:
        csv_file_path = os.path.join(f"noise_plots_brevitas/gaussian_noise/{model_name}", "raw_data_proportional_all_layers.csv")

    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Sigma Value", "Average Accuracy"])

        # Create a list to store the test accuracies for all layers
        all_test_accs = []

        random.seed(datetime.now().timestamp())

        print("Random Seed Generated")

        for sigma in sigma_vector:
            # Add noise to the model for all layers
            noisy_models = add_gaussian_noise_to_model_brevitas(
                model, layer_names, sigma, num_perturbations, analog_noise_type)
            accuracies = []

            for noisy_model in noisy_models:
                noisy_model.to(device)
                accuracies.append(test(noisy_model, test_quantized_loader, device))

            avg_accuracy = sum(accuracies) / len(accuracies)
            all_test_accs.append(avg_accuracy)

            # Write the raw data to the CSV file
            writer.writerow([sigma, avg_accuracy])

            if analog_noise_type:
                print("Independent: Sigma Value: {}, Average Accuracy: {}%".format(sigma, avg_accuracy))
            else:
                print("Proportional: Sigma Value: {}, Average Accuracy: {}%".format(sigma, avg_accuracy))

        # Plot the test accuracies as a function of the standard deviation for all layers
        plt.plot(sigma_vector, all_test_accs, label='Average Accuracy at Different Perturbation Levels')

        plt.xlabel('Standard Deviation')
        plt.ylabel('Test Accuracy')
        plt.title('Effect of Noise on Test Accuracy (All Layers)')
        plt.legend()

        if analog_noise_type:
            plt.savefig(f"noise_plots_brevitas/gaussian_noise/{model_name}/all_layers_independent.png")
        else:
            plt.savefig(f"noise_plots_brevitas/gaussian_noise/{model_name}/all_layers_proportional.png")

        plt.show()
        plt.clf()


## Gaussian Noise Testing

layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'fc1', 'fc2']

## Gaussian Noise
sigma_vector = np.linspace(0, 0.008, 15)

#gaussian_noise_plots_brevitas(perturbations, layer_names, sigma_vector, model, device, val_quantized_loader, 0, model_name)
#gaussian_noise_plots_brevitas(perturbations, layer_names, sigma_vector, model, device, val_quantized_loader, 1, model_name)
gaussian_noise_plots_brevitas_all(perturbations, layer_names, sigma_vector, model, device, val_quantized_loader, 0, model_name)
#gaussian_noise_plots_brevitas_all(perturbations, layer_names, sigma_vector, model, device, val_quantized_loader, 1, model_name)
