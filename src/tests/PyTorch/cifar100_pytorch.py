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
    08 December 2023

Last Modified:
    17 January 2024

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
current_script_path = Path(__file__).parent.parent

# Get the parent directory of the current script
parent_directory = current_script_path.parent

# Add the parent directory to the Python path
sys.path.append( str( parent_directory ) )

# Now, you can import from the parent directory
from utils import noise_injection_pytorch_funcs as noise_funcs     # import pytorch functions
from utils import models
from utils import setup

## Initialize Variables

# Data Augmentation
crop_size = 32
padding = 4
mean_vector_training = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std_vector_training = [.2673342858792401, 0.2564384629170883, 0.27615047132568404]
mean_vector_transform = [0.5088964127604166, 0.48739301317401956, 0.44194221124387256]
std_vector_transform = [0.2682515741720801, 0.2573637364478126, 0.2770957707973042]

setup.setup_test( crop_size, padding, mean_vector_training,
                  std_vector_training, mean_vector_transform,
                  std_vector_transform )


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

print("-----------------------------------------------------")
print("---------- Beginning Bit Error Rate Testing ---------")
print("-----------------------------------------------------")

## BER Noise Testing
ber_vals = np.linspace( 1e-5, 0.01, 15 )

print( "\n-----------------------------------------------------")
print( "------- Beginning PyTorch BER Noise Injection ------")
print( "----------------- Individual Layers -----------------")
print( "-----------------------------------------------------")
noise_funcs.ber_noise_plot_pytorch( perturbations, layer_names, ber_vals, model, device, val_quantized_loader, model_name )

print( "\n\n-----------------------------------------------------")
print( "------- Beginning Brevitas BER Noise Injection ------")
print( "------------------ Multiple Layers ------------------")
print( "-----------------------------------------------------")
noise_funcs.ber_noise_plot_multiple_layers_pytorch( perturbations, layer_names, ber_vals, model, device, val_quantized_loader, model_name )

print("\n\n-----------------------------------------------------")
print("---------- Beginning Gaussian Noise Testing ---------")
print("-----------------------------------------------------")

## Gaussian Noise Testing
# Gaussian sigma vector. Retains most values from BER but changes the sigma_vector.
sigma_vector_prop = np.linspace( 0, 0.10, 15 )
sigma_vector_ind = np.linspace( 0, 0.005, 15 )

print( "\n-----------------------------------------------------" )
print( "---- Beginning Brevitas Gaussian Noise Injection ----" )
print( "----------------- Individual Layers -----------------" )
print( "-------------------- Proportional -------------------" )
print( "-----------------------------------------------------" )
noise_funcs.gaussian_noise_plot_pytorch( perturbations, layer_names, sigma_vector_prop, model, device, val_quantized_loader, 0, model_name )

print( "\n\n-----------------------------------------------------" )
print( "---- Beginning Brevitas Gaussian Noise Injection ----" )
print( "----------------- Individual Layers -----------------" )
print( "-------------------- Independent --------------------" )
print( "-----------------------------------------------------" )
noise_funcs.gaussian_noise_plot_pytorch( perturbations, layer_names, sigma_vector_ind, model, device, val_quantized_loader, 1, model_name )

print( "\n\n-----------------------------------------------------" )
print( "---- Beginning Brevitas Gaussian Noise Injection ----" )
print( "------------------- Multiple Layers -----------------" )
print( "-------------------- Proportional -------------------" )
print( "-----------------------------------------------------" )
noise_funcs.gaussian_noise_plot_multiple_layers_pytorch( perturbations, layer_names, sigma_vector_prop, model, device, val_quantized_loader, 0, model_name )

print( "\n\n-----------------------------------------------------" )
print( "---- Beginning Brevitas Gaussian Noise Injection ----" )
print( "------------------- Multiple Layers -----------------" )
print( "-------------------- Independent --------------------" )
print( "-----------------------------------------------------" )
noise_funcs.gaussian_noise_plot_multiple_layers_pytorch( perturbations, layer_names, sigma_vector_ind, model, device, val_quantized_loader, 1, model_name )
