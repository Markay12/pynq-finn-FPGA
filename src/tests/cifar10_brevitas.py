"""
Cifar10 Brevitas Noise Analysis Testing
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
    transforms.RandomCrop( 32, padding=4 ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize( mean=[ 0.5, 0.5, 0.5 ], std=[ 0.5, 0.5, 0.5 ] )
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize( mean=[ 0.5, 0.5, 0.5 ], std=[ 0.5, 0.5, 0.5 ] )
])

print("-----------------------------------------------------")
print("------------ Applying Data Augmentation -------------")
print("-----------------------------------------------------")

# Apply data augmentation to the training dataset
train_set = torchvision.datasets.CIFAR10( root='../data', train=True, download=True, transform=train_transform )

# Use the validation transform for the validation dataset
val_set = torchvision.datasets.CIFAR10( root='../data', train=False, download=True, transform=val_transform )

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
batch_size = 1000

# Create a DataLoader for a training dataset with a batch size of 1000
train_quantized_loader = DataLoader( train_set, batch_size=batch_size )
test_quantized_loader = DataLoader( val_set, batch_size=batch_size )

# CIFAR10CNN: A quantized CNN model for CIFAR-10 dataset classification using Brevitas.
# Comprises QuantConv2d and QuantReLU layers with max pooling and fully connected 
# layers for efficient image classification.

class CIFAR10CNN(nn.Module):

    def __init__(self):
        super( CIFAR10CNN, self ).__init__()
        self.quant_inp = qnn.QuantIdentity( bit_width=4, return_quant_tensor=True )

        self.layer1 = qnn.QuantConv2d( 3, 32, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias )
        self.relu1 = qnn.QuantReLU( bit_width=4, return_quant_tensor=True )

        self.layer2 = qnn.QuantConv2d( 32, 32, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias )
        self.relu2 = qnn.QuantReLU( bit_width=4, return_quant_tensor=True )

        self.layer3 = qnn.QuantConv2d( 32, 64, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias )
        self.relu3 = qnn.QuantReLU( bit_width=4, return_quant_tensor=True )

        self.layer4 = qnn.QuantConv2d( 64, 64, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias )
        self.relu4 = qnn.QuantReLU( bit_width=4, return_quant_tensor=True )

        self.layer5 = qnn.QuantConv2d( 64, 64, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias )
        self.relu5 = qnn.QuantReLU( bit_width=4, return_quant_tensor=True )

        self.fc1 = qnn.QuantLinear( 64 * 8 * 8, 512, bias=True, weight_bit_width=4, bias_quant=Int32Bias )
        self.relu6 = qnn.QuantReLU( bit_width=4, return_quant_tensor=True )

        self.fc2 = qnn.QuantLinear( 512, 10, bias=True, weight_bit_width=4, bias_quant=Int32Bias )

    def forward( self, x ):
        x = self.quant_inp( x )
        x = self.relu1( self.layer1( x ) )
        x = self.relu2( self.layer2( x ) )
        x = F.max_pool2d( x, 2 )

        x = self.relu3( self.layer3( x ) )
        x = self.relu4( self.layer4( x ) )
        x = F.max_pool2d( x, 2 )

        x = self.relu5( self.layer5( x ) )

        x = x.view( x.size( 0 ), -1 )

        x = self.relu6( self.fc1( x ) )
        x = self.fc2( x )

        return x

print("-----------------------------------------------------")
print("---- Model, Optimizer and Criteria Initialized ------")
print("-----------------------------------------------------")
    
# Initialize the model, optimizer, and criterion
model = CIFAR10CNN().to( device )
optimizer = torch.optim.AdamW( model.parameters(), lr=0.001 )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, patience=5, verbose=True )
criterion = nn.CrossEntropyLoss()

num_epochs = 80  # Total number of training epochs (full passes over the dataset).
best_test_accuracy = 0  # Keep track of the highest test accuracy achieved.
patience = 8  # Number of epochs to wait for improvement in accuracy before early stopping.
no_improvement_counter = 0  # Counter to track epochs without improvement in accuracy.

# Create an instance of neural network model
model = CIFAR10CNN().to( device )
model_name = input( "Name of the Model to Test: " )

# Load the saved state dictionary from file
state_dict = torch.load( f'C:\\Users\\ashin\\source\\repos\\Cifar10_Pytorch_NoiseAnalysis\\Cifar10_Pytorch_NoiseAnalysis\\pynq-finn-FPGA\\noise_weight_analysis\\Cifar10\\Brevitas\\main_testing\\models\\{model_name}.pth', map_location=device )

# Load the state dictionary into the model
model.load_state_dict( state_dict )

print("-----------------------------------------------------")
print("------------- Printing Layer Shape Sizes ------------")
print("-----------------------------------------------------")

# Test shapes
print(  model.layer1.weight.shape )
print(  model.layer2.weight.shape )
print(  model.fc1.weight.shape )

print("-----------------------------------------------------")
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
layer_names = [ 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'fc1', 'fc2' ]
layer_combinations = [ [ 'layer1', 'layer2', 'layer3', 'layer4', 'layer5' ] ]
p_values = [ 1, 0.5, 0.25 ]
gamma_values = np.linspace( 0.001, 0.1, 2 )
sigma = np.linspace( 0.0, 0.2, 2 )

print("-----------------------------------------------------")
print("---------- Beginning Bit Error Rate Testing ---------")
print("-----------------------------------------------------")

## BER Noise Testing
ber_vals = np.linspace( 1e-5, 0.01, 15 )

noise_funcs.ber_noise_plot_brevitas( perturbations, layer_names, ber_vals, model, device, test_quantized_loader, model_name )
noise_funcs.ber_noise_plot_brevitas_multiple_layers( perturbations, layer_names, ber_vals, model, device, test_quantized_loader, model_name )

print("-----------------------------------------------------")
print("---------- Beginning Gaussian Noise Testing ---------")
print("-----------------------------------------------------")

## Gaussian Noise Testing
# Gaussian sigma vector. Retains most values from BER but changes the sigma_vector.
sigma_vector = np.linspace(0, 0.05, 15)

noise_funcs.gaussian_noise_plots_brevitas( perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, 0, model_name )
noise_funcs.gaussian_noise_plots_brevitas( perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, 1, model_name )
noise_funcs.gaussian_noise_plots_brevitas_all( perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, 0, model_name )
noise_funcs.gaussian_noise_plots_brevitas_all( perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, 1, model_name )
