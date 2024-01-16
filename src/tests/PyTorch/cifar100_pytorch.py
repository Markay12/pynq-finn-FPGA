"""
Cifar100 Brevitas Noise Analysis Testing
-------------------------

Description:
    - 

Notes:
    - 
    
Author(s):
    Mark Ashinhust

Created on:
    08 December 2024

Last Modified:
    08 December 2024

"""

print("-----------------------------------------------------")
print("---------- Importing Modules and Functions ----------")
print("-----------------------------------------------------")

## Import statements
import brevitas.nn as qnn
from brevitas.quant import Int32Bias
import datetime
import os
import numpy as np
import random
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

## Import utility functions for noise testing
# Get the directory of the current script
current_script_path = Path(__file__).parent

# Get the parent directory of the current script
parent_directory = current_script_path.parent

# Add the parent directory to the Python path
sys.path.append( str( parent_directory ) )

# Now, you can import from the parent directory
from utils import noise_injection_brevitas_funcs as noise_funcs
from utils import models

print("-----------------------------------------------------")
print("-------------- Locating Targeted Device -------------")
print("-----------------------------------------------------")

# Locate the device that can be used for noise analysis and testing
# Target this device 
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
print( "Target device: " + str (device ) )

print("-----------------------------------------------------")
print("------- Defining Data Augmentation Transforms -------")
print("-----------------------------------------------------")

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

print("-----------------------------------------------------")
print("------------ Applying Data Augmentation -------------")
print("-----------------------------------------------------")

# Apply data augmentation to the training dataset
train_set = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)

# Use the validation transform for the validation dataset
val_set = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=val_transform)

print("-----------------------------------------------------")
print("--------------- Creating Data Loaders ---------------")
print("-----------------------------------------------------")

# Create the data loaders
train_loader = torch.utils.data.DataLoader( train_set, batch_size=128, shuffle=True, num_workers=4 )
val_loader = torch.utils.data.DataLoader( val_set, batch_size=128, shuffle=False, num_workers=4 )

## Data Loader
# Using PyTorch dataloader we can create a convenient iterator over the dataset that returns batches of data, 
# rather than requiring manual batch creation.

# set batch size
batch_size = 256

train_set = torchvision.datasets.CIFAR100(root='./data/', train=True, download=True, transform=train_transform)
val_set = torchvision.datasets.CIFAR100(root='./data/', train=False, download=True, transform=val_transform)

train_quantized_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_quantized_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

a = next(iter(train_loader))
print(a[0].size())
print(len(train_set))
print("Samples in each set: train = %d, test = %s" % (len(train_set), len(train_loader)))
print("Shape of one input sample: " + str(train_set[0][0].shape))

count = 0

print("\nDataset Shape:\n-------------------------")
for x, y in train_loader:
    print("Input shape for 1 batch: " + str(x.shape))
    print("Label shape for 1 batch: " + str(y.shape))
    count += 1
    if count == 1:
        break

print("-----------------------------------------------------")
print("---- Model, Optimizer and Criteria Initialized ------")
print("-----------------------------------------------------")
    
# Initialize the model, optimizer, and criterion
model     = models.CIFAR100CNN_5Blocks().to( device )

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0009)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=11, factor=0.1, verbose=True)
criterion = nn.CrossEntropyLoss()

num_epochs = 80             # Total number of training epochs (full passes over the dataset).
best_test_accuracy = 0      # Keep track of the highest test accuracy achieved.
patience = 8                # Number of epochs to wait for improvement in accuracy before early stopping.
no_improvement_counter = 0  # Counter to track epochs without improvement in accuracy.

model_name = input( "Name of the Model to Test: " )
state_dict = torch.load( f'{os.getcwd()}/trained_models/{model_name}.pth', map_location=device )


# Load the state dictionary into the model
model.load_state_dict( state_dict )

print("-----------------------------------------------------")
print("------------- Printing Layer Shape Sizes ------------")
print("-----------------------------------------------------")

# Test shapes
print( "Convolutional Layer 1 Weight Shape: ",   model.conv1.conv.weight.shape )
print( "Convolutional Layer 1 Bias: ",           model.conv1.conv.bias )
print( "Fully Connected Layer 1 Weight Shape: ", model.fc1.weight.shape )


print("\n-----------------------------------------------------")
print("--------------- Generating Random Seed --------------")
print("-----------------------------------------------------")

## Testing Models
# Generate a seed based on the current date and time
current_time = datetime.datetime.now()
seed = int( current_time.timestamp() )

# Set the seed for the random module
random.seed( seed )

perturbations = 15 # Value does not change between tests

# set the layer names, combinations, probability, gamma and sigma values for BER testing
layer_names = [ 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'fc1', 'fc2' ]
layer_combinations = [ [ 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'fc1', 'fc2' ] ]
p_values = [ 1, 0.5, 0.25 ]
gamma_values = np.linspace( 0.001, 0.1, 2 )
sigma = np.linspace( 0.0, 0.2, 2 )

print("-----------------------------------------------------")
print("---------- Beginning Bit Error Rate Testing ---------")
print("-----------------------------------------------------")

## BER Noise Testing
ber_vals = np.linspace( 1e-5, 0.01, 15 )

print( "\n-----------------------------------------------------")
print( "------- Beginning Brevitas BER Noise Injection ------")
print( "----------------- Individual Layers -----------------")
print( "-----------------------------------------------------")
#noise_funcs.ber_noise_plot_brevitas( perturbations, layer_names, ber_vals, model, device, val_quantized_loader, model_name )

print( "\n\n-----------------------------------------------------")
print( "------- Beginning Brevitas BER Noise Injection ------")
print( "------------------ Multiple Layers ------------------")
print( "-----------------------------------------------------")
noise_funcs.ber_noise_plot_multiple_layers_brevitas( perturbations, layer_names, ber_vals, model, device, val_quantized_loader, model_name )

print("\n\n-----------------------------------------------------")
print("---------- Beginning Gaussian Noise Testing ---------")
print("-----------------------------------------------------")

## Gaussian Noise Testing
# Gaussian sigma vector. Retains most values from BER but changes the sigma_vector.
sigma_vector = np.linspace(0, 0.05, 15)

print( "\n-----------------------------------------------------" )
print( "---- Beginning Brevitas Gaussian Noise Injection ----" )
print( "----------------- Individual Layers -----------------" )
print( "-------------------- Proportional -------------------" )
print( "-----------------------------------------------------" )
noise_funcs.gaussian_noise_plots_brevitas( perturbations, layer_names, sigma_vector, model, device, val_quantized_loader, 0, model_name )

print( "\n\n-----------------------------------------------------" )
print( "---- Beginning Brevitas Gaussian Noise Injection ----" )
print( "----------------- Individual Layers -----------------" )
print( "-------------------- Independent --------------------" )
print( "-----------------------------------------------------" )
noise_funcs.gaussian_noise_plots_brevitas( perturbations, layer_names, sigma_vector, model, device, val_quantized_loader, 1, model_name )

print( "\n\n-----------------------------------------------------" )
print( "---- Beginning Brevitas Gaussian Noise Injection ----" )
print( "------------------- Multiple Layers -----------------" )
print( "-------------------- Proportional -------------------" )
print( "-----------------------------------------------------" )
noise_funcs.gaussian_noise_plots_brevitas_all( perturbations, layer_names, sigma_vector, model, device, val_quantized_loader, 0, model_name )

print( "\n\n-----------------------------------------------------" )
print( "---- Beginning Brevitas Gaussian Noise Injection ----" )
print( "------------------- Multiple Layers -----------------" )
print( "-------------------- Independent --------------------" )
print( "-----------------------------------------------------" )
noise_funcs.gaussian_noise_plots_brevitas_all( perturbations, layer_names, sigma_vector, model, device, val_quantized_loader, 1, model_name )
