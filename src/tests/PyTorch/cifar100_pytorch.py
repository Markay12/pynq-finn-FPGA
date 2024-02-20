"""
CIFAR100 PyTorch Noise Analysis Testing
----------------------------------------

Description:
    This script is designed for conducting noise analysis testing on CIFAR100 neural network models 
    implemented with PyTorch. It focuses on evaluating the robustness of the models under 
    different noise conditions, specifically Bit Error Rate (BER) and Gaussian noise. The script 
    orchestrates the entire testing process, from initializing the model and loading the dataset, 
    to applying noise perturbations and evaluating the impact on model performance. The noise 
    analysis aims to assess the susceptibility of neural networks to hardware-induced errors, 
    simulating scenarios that might be encountered in real-world deployments.

Notes:
    - The script covers both individual layer noise injection and multiple layers for a comprehensive analysis.
    - It employs a range of BER and Gaussian noise levels to thoroughly test the model's robustness.
    - The testing framework is built for flexibility, allowing easy adaptation to different models or datasets.
    
Author(s):
    Mark Ashinhust

Created on:
    08 December 2023

Last Modified:
    18 January 2024

Usage:
    To use this script, ensure that the CIFAR100 dataset and the required PyTorch models are 
    accessible. Modify the layer names and noise parameters as needed for specific testing 
    requirements. Run the script in an environment where PyTorch is installed. The results of 
    the noise tests will be printed out, providing insights into the model's performance under 
    various noise conditions.
"""


print("-----------------------------------------------------")
print("---------- Importing Modules and Functions ----------")
print("-----------------------------------------------------")

## Import statements
import numpy as np
from pathlib import Path
import sys
import torch.nn as nn

## Import utility functions for noise testing
# Get the directory of the current script
current_script_path = Path(__file__).parent.parent

# Get the parent directory of the current script
parent_directory = current_script_path.parent

# Add the parent directory to the Python path
sys.path.append( str( parent_directory ) )

# Now, you can import from the parent directory
from utils import noise_injection_pytorch_funcs as test_start     # import pytorch functions
from utils import setup

## Initialize Variables

# Data Augmentation
crop_size = 32
padding = 4
mean_vector_training = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std_vector_training = [.2673342858792401, 0.2564384629170883, 0.27615047132568404]
mean_vector_transform = [0.5088964127604166, 0.48739301317401956, 0.44194221124387256]
std_vector_transform = [0.2682515741720801, 0.2573637364478126, 0.2770957707973042]

## Begin Test Setup
device, model, val_quantized_loader, model_name = setup.setup_test( crop_size, padding, mean_vector_training, std_vector_training, mean_vector_transform,
                                                                    std_vector_transform )


setup.gen_rand_seed()

## Intialize Test Structure
perturbations = 15

layer_names = setup.extract_layer_names( model )
print( layer_names ) 

# BER Vector
ber_vals = np.linspace( 1e-5, 0.01, 15 )

# Gaussian Vector
# Gaussian sigma vector. Retains most values from BER but changes the sigma_vector.
sigma_vector_prop = np.linspace( 0, 0.10, 15 )
sigma_vector_ind = np.linspace( 0, 0.05, 15 )

test_start.select_test( perturbations, layer_names, ber_vals, model, device, 
                        val_quantized_loader, model_name, sigma_vector_prop, sigma_vector_ind )
