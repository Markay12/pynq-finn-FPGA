"""
Description:
This script performs the following functionalities:

1. **Imports**: Necessary libraries and modules such as PyTorch, torchvision, numpy, and custom functions for the CIFAR100 dataset and BER (Bit Error Rate) noise testing are imported.

2. **Data Loading**:
   - Data augmentation and normalization transformations are defined for training and validation datasets.
   - CIFAR100 datasets are loaded with the specified transformations.
   - PyTorch data loaders are initialized to enable efficient iteration over the datasets in mini-batches.

3. **Model Initialization**:
   - A pre-trained model state dictionary is loaded from a specified path.
   - The CIFAR100CNN_5Blocks neural network model is initialized and set to run on the available hardware (either GPU or CPU).
   - An optimizer (AdamW) with specified learning rate parameters and a learning rate scheduler are initialized.
   - Cross-entropy loss function is set up for training the model.

4. **Model Testing**:
   - The model undergoes testing using BER (Bit Error Rate) noise perturbations.
   - Different layers of the model and a range of BER values are specified.
   - The `ber_noise_plot_brevitas` function is used to perform the noise testing and visualize the results.

The main purpose of this script is to load the CIFAR100 dataset, initialize a specific neural network model, and evaluate its robustness against different levels of BER noise perturbations.

Date: 24 August 2023
"""


#############################
#####      Imports      #####
#############################

# Import necessary modules and libraries
import importlib
import sys
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np

# Import custom functions and model components
from ber_functions import test, add_digital_noise, add_digital_noise_to_model_brevitas, ber_noise_plot_brevitas
from model_functions import CommonUintActQuant, CommonIntWeightPerTensorQuant, ConvBlock, CIFAR100CNN_4Blocks, CIFAR100CNN_5Blocks

# Check and set the device (either CUDA for GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Target device: " + str(device))

#############################
#####   Data Loading    #####
#############################

# Define data augmentation and normalization transforms for training and validation datasets
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

# Load training and validation datasets with the specified transforms
train_set = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
val_set = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=val_transform)
print("\nTrain and Validation set loaded\n")

# Define data loaders for the datasets
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)

# Load quantized versions of the training and validation datasets
train_quantized_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4)
val_quantized_loader = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=4)

# Print some information about the loaded datasets
a = next(iter(train_loader))
print(a[0].size())
print(len(train_set))
print("Samples in each set: train = %d, test = %s" % (len(train_set), len(train_loader)))
print("Shape of one input sample: " + str(train_set[0][0].shape))
print("\nDataset Shape:\n-------------------------")
for x, y in train_loader:
    print("Input shape for 1 batch: " + str(x.shape))
    print("Label shape for 1 batch: " + str(y.shape))
    break

#############################
##### Model Initialization#####
#############################

# Load a saved model state dictionary
state_dict = torch.load(f'//home//mashinhu//Desktop//Cifar100Testing//tested_models//best_model__6.0_6.0_.pth', map_location=device)
print("\nState Dict Loaded\n")

# Initialize the CIFAR100CNN_5Blocks model, optimizer, and loss function
model = CIFAR100CNN_5Blocks().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0009)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=11, factor=0.1, verbose=True)
criterion = nn.CrossEntropyLoss()

# Load the state dictionary into the initialized model
model.load_state_dict(state_dict)

# Ask the user for a model name
model_name = input("Name for the model: ")

# Print shapes of some model parameters for debugging purposes
print(model.conv1.conv.weight.shape)
print(model.conv1.conv.bias)
print(model.conv2.conv.weight.shape)
print(model.fc1.weight.shape)

#############################
#####   Model Testing    #####
#############################

print("\nBeginning to test perturbations\n")
perturbations = 20  # Number of perturbations to test

# BER Noise Testing
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'fc1', 'fc2']
ber_vals = np.linspace(1e-5, 0.01, 15)

# Test the BER noise using the provided function and plot the results
ber_noise_plot_brevitas(perturbations, layer_names, ber_vals, model, device, val_quantized_loader, model_name)
ber_noise_plot_brevitas_multiple_layers(perturbations, layer_names, ber_vals, model, device, val_quantized_loader, model_name)
