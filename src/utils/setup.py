"""
Setup for Testing
-------------------------

Description:
    This module provides a comprehensive setup for testing neural networks, particularly focusing on 
    noise injection and performance analysis in PyTorch models. It includes functions for data augmentation, 
    data loader creation, model setup, and various utilities to facilitate neural network testing in a 
    quantized environment. The emphasis is on ease of use and integration into existing testing workflows.

Functions:
    locate_hw_target()
    augment_data()
    create_data_loaders()
    extract_layer_names()
    model_criteria()
    load_dictionary()
    setup_test()
    gen_rand_seed()
    print_header()

Usage:
    These utilities are intended to be imported and used in data processing or
    simulation scripts. Example of use:
    
        from noise_injection import function1_name

        modified_data = function1_name(data)

Notes:
    - These functions are optimized for performance and accuracy. Most of these functions are tailored specifically to CIFAR100.
      However, this should be able to be changed and future functionality can be introduced to make this a parameter.
    
Author(s):
    Mark Ashinhust

Created on:
    17 January 2024

Last Modified:
    18 January 2024

"""

import datetime
import os
import numpy as np
from pathlib import Path
import random
import sys
import time
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# Current script path
current_path = Path( __file__ ).parent

# Get parent directory
parent_directory = current_path

print( parent_directory )

# Add parent to the Python path
sys.path.append( str( parent_directory ) )

import models

"""
Function Name: locate_hw_target()

Parameters:
    None

Function Process:
    - The function identifies the most suitable hardware target for running neural network models.
    - It checks if a CUDA-capable GPU is available. If so, it selects the GPU as the target device.
    - If a GPU is not available, it falls back to using the CPU.
    - The chosen device (either GPU or CPU) is then printed to the console for confirmation.
    - The function is useful for ensuring that models are run on the most efficient hardware 
      available, particularly important for performance-intensive tasks like training or 
      evaluating neural networks.

Return:
    - torch.device: The selected hardware device (CUDA GPU or CPU) for running the models. 
                    This device object can be used in PyTorch to allocate tensors directly 
                    on the chosen device.
"""

def locate_hw_target():

    # print statement for debug
    print_header( "Locating Targeted Device" )

    # Locate the device that can be used for noise analysis and testing
    # Target this device 
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    print( "Target device: " + str (device ) )

    # return whether using CUDA or CPU
    return device

"""
Function Name: augment_data()

Parameters:
    1. crop_size
        Description:    Size of the cropped image.
        Type:           Integer
        Details:        This value specifies the dimensions to which images will be cropped 
                        during the augmentation process.

    2. padding_size
        Description:    Size of padding around the image before cropping.
        Type:           Integer
        Details:        Padding is added to the images before the cropping step. This helps in 
                        creating variations in the dataset by altering the area of the image 
                        that is cropped.

    3. mean_vector_train
        Description:    Mean values for normalization in training data.
        Type:           List of Floats
        Details:        These values are used for normalizing the pixel values of the training 
                        images, aiding in model convergence during training.

    4. std_vector_train
        Description:    Standard deviation values for normalization in training data.
        Type:           List of Floats
        Details:        Standard deviation values used alongside the mean for normalization of 
                        training images.

    5. mean_vector_transform
        Description:    Mean values for normalization in validation data.
        Type:           List of Floats
        Details:        Similar to mean_vector_train but applied to validation dataset for 
                        consistent data preprocessing.

    6. std_vector_transform
        Description:    Standard deviation values for normalization in validation data.
        Type:           List of Floats
        Details:        Similar to std_vector_train but applied to validation dataset for 
                        consistent data preprocessing.

Function Process:
    - The function defines a series of data augmentation and transformation steps, including
      random cropping, horizontal flipping, and normalization.
    - These transformations are applied to both training and validation datasets but with 
      different normalization parameters for each.
    - The augmented and transformed datasets are then loaded using the CIFAR100 dataset from 
      torchvision.

Return:
    - train_set: The augmented and transformed training dataset.
    - val_set: The transformed validation dataset.
    - train_transform: The transformation applied to the training dataset.
    - val_transform: The transformation applied to the validation dataset.

Note:
    This function is specifically designed for the CIFAR100 dataset but can be adapted for 
    other datasets by modifying the dataset loading step.
"""

def augment_data( crop_size, padding_size, mean_vector_train, std_vector_train, mean_vector_transform, std_vector_transform ):

    # print statement for debug
    print_header( "Defining Data Augmentation Transforms" )

    # Define data augmentation transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop( crop_size, padding = padding_size ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize( mean = mean_vector_train, std = std_vector_train )
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize( mean = mean_vector_transform, std = std_vector_transform )
    ])

    print_header( "Applying Data Augmentation" )

    # Apply data augmentation to the training dataset
    train_set = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)

    # Use the validation transform for the validation dataset
    val_set = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=val_transform)

    return train_set, val_set, train_transform, val_transform

"""
Function Name: create_data_loaders()

Parameters:
    1. train_set
        Description:    The training dataset after applying augmentation and transformations.
        Type:           torchvision.datasets
        Details:        This dataset is used to create the training data loader.

    2. val_set
        Description:    The validation dataset after applying transformations.
        Type:           torchvision.datasets
        Details:        This dataset is used to create the validation data loader.

    3. train_transform
        Description:    Transformations applied to the training dataset.
        Type:           torchvision.transforms
        Details:        Not directly used in this function, but provided for consistency 
                        and potential future use.

    4. val_transform
        Description:    Transformations applied to the validation dataset.
        Type:           torchvision.transforms
        Details:        Not directly used in this function, but provided for consistency 
                        and potential future use.

    5. batch_size
        Description:    The size of each batch of data.
        Type:           Integer
        Details:        Determines how many samples are processed together in one iteration 
                        of the data loader.

Function Process:
    - The function initializes data loaders for both the training and validation datasets.
    - It sets up the batch size, shuffling, and number of worker threads for efficient 
      data loading and processing.
    - The function also includes debug prints to provide information about the shape and 
      size of the datasets and the batches.

Return:
    - train_quantized_loader: Data loader for the training dataset.
    - val_quantized_loader: Data loader for the validation dataset.

Note:
    The function is tailored for the CIFAR100 dataset but can be adapted for other datasets 
    by modifying the dataset-specific parameters.
"""

def create_data_loaders( train_set, val_set, train_transform, val_transform, batch_size ):

    # print statement for debug
    print_header( "Creating Data Loaders" )

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader( train_set, batch_size=128, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_set, batch_size=128, shuffle=False, num_workers=4 )

    ## Data Loader
    # Using PyTorch dataloader we can create a convenient iterator over the dataset that returns batches of data, 
    # rather than requiring manual batch creation.

    # set batch size
    batch_size = batch_size

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

    return train_quantized_loader, val_quantized_loader

"""
Function Name: extract_layer_names()

Parameters:
    1. model

Function Process:
    - 

Return:
    - 

Note:
    
"""

def extract_layer_names( model ):
	layer_names = []
     
	for name, module in model.named_modules():
          
		if isinstance( module, (nn.Conv2d, nn.Linear )):
			
			cleaned_name = name.replace( '.conv', '' )
			layer_names.append( cleaned_name )

	return layer_names

"""
Function Name: model_criteria()

Parameters:
    1. device
        Description:    The target device for model computation (CPU or GPU).
        Type:           torch.device
        Details:        Determines where the model and computations will be allocated, 
                        impacting performance and efficiency.

Function Process:
    - The function initializes a neural network model, along with its optimizer, learning 
      rate scheduler, and loss criterion.
    - It sets the model to the specified device (GPU or CPU) for optimized computation.
    - The function prompts the user to input the name of the model for testing, and loads 
      the corresponding pre-trained state dictionary.
    - Several training parameters such as number of epochs, patience for early stopping, and 
      initial best test accuracy are defined within the function.
    - It is designed to provide a comprehensive setup for model training and evaluation, 
      making it easier to experiment with different configurations.

Return:
    - model: The neural network model, set to the specified device.
    - state_dict: The state dictionary loaded from the saved model file.
    - optimizer: The optimizer configured for the model.
    - scheduler: Learning rate scheduler for adjusting the learning rate during training.
    - criterion: Loss function used for model training.
    - model_name: Name of the model as input by the user.

Note:
    This function is tailored for the CIFAR100CNN_5Blocks model but can be adapted for other 
    models by modifying the model initialization and parameter settings.
"""

def model_criteria( device ):

    # print for debug
    print_header( "Model, Optimizer, and Criteria Initialized" )

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

    return model, state_dict, optimizer, scheduler, criterion, model_name

"""
Function Name: load_dictionary()

Parameters:
    1. model
        Description:    The neural network model into which the state dictionary will be loaded.
        Type:           torch.nn.Module
        Details:        This is the pre-initialized model whose parameters are to be set 
                        according to the provided state dictionary.

    2. state_dict
        Description:    The state dictionary containing model weights and biases.
        Type:           Dict
        Details:        This dictionary holds the trained parameters of the model, typically 
                        obtained from a previously saved model state.

Function Process:
    - The function loads the provided state dictionary into the given model.
    - This operation updates the model's parameters (weights and biases) to match those 
      stored in the state dictionary.
    - After loading the state dictionary, the function prints out the shapes of the weights 
      and biases of certain layers for verification and debugging purposes. 
    - This can help in ensuring that the model parameters are loaded correctly and the 
      model architecture matches the state dictionary.

Return:
    None

Note:
    This function is a utility for loading a pre-trained model state. It is particularly 
    useful in scenarios where a model trained in one session needs to be reloaded for 
    further use or evaluation in another session.
"""

def load_dictionary( model, state_dict ):

    # Load the state dictionary into the model
    model.load_state_dict( state_dict )

    print_header( "Printing Layer Shape Sizes" )

    # Test shapes
    print( "Convolutional Layer 1 Weight Shape: ",   model.conv1.conv.weight.shape )
    print( "Convolutional Layer 1 Bias: ",           model.conv1.conv.bias )
    print( "Fully Connected Layer 1 Weight Shape: ", model.fc1.weight.shape )

"""
Function Name: setup_test()

Parameters:
    1. crop_size, padding_size, mean_vector_train, std_vector_train, mean_vector_transform, std_vector_transform
        Description:    Parameters required for data augmentation and transformation.
        Type:           Various (Integer and Lists of Floats)
        Details:        These parameters are used to define how the input data for the model 
                        should be augmented and transformed.

Function Process:
    - This function serves as the main setup procedure for neural network testing.
    - It begins by locating the hardware target (CPU or GPU) for model computation.
    - It then augments and transforms the training and validation datasets.
    - Following this, it creates data loaders for both datasets.
    - The function initializes the neural network model and loads the pretrained model weights.
    - Finally, it returns the device, model, validation data loader, and model name for further use.

Return:
    - device: The computation device (CPU or GPU).
    - model: The initialized neural network model.
    - val_quantized_loader: DataLoader for the validation dataset.
    - model_name: Name of the model for testing.
"""

def setup_test( crop_size, padding_size, mean_vector_train, std_vector_train, mean_vector_transform, std_vector_transform ):

    device = locate_hw_target()
    
    train_set, val_set, train_transform, val_transform = augment_data(crop_size, padding_size, mean_vector_train, std_vector_train, mean_vector_transform, std_vector_transform)
    train_quantized_loader, val_quantized_loader = create_data_loaders( train_set, val_set, train_transform, val_transform, 256 )
    model, state_dict, optimizer, scheduler, criterion, model_name = model_criteria( device )
    
    load_dictionary( model, state_dict )

    return device, model, val_quantized_loader, model_name

"""
Function Name: gen_rand_seed()

Parameters:
    None

Function Process:
    - Generates a random seed based on the current date and time.
    - This seed is used to initialize the random module, ensuring that any random operation 
      within the testing models is reproducible.
    - Especially useful in scenarios where consistent behavior is required across different 
      runs for comparison or debugging purposes.

Return:
    None
"""

def gen_rand_seed():

    time.sleep(1)

    # print for debug
    print_header( "Generating Random Seed" )

    current_time = datetime.datetime.now()
    seed = int(current_time.timestamp())
    print(f"Random seed: {seed}")

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # If you are using other libraries that depend on randomness, set their seeds as well.
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

"""
Function Name: print_header()

Parameters:
    1. title
        Description:    The main title for the header.
        Type:           String
        Details:        This title is printed as a part of the header for better readability 
                        and organization of console output.

    2. subtitle (Optional)
        Description:    An additional subtitle for more detailed headers.
        Type:           String
        Details:        Provides additional context or categorization within the header.

    3. multiple_layers (Optional)
        Description:    Flag to indicate if the header is for multiple layers.
        Type:           Boolean
        Details:        When set to True, an additional line indicating 'Multiple Layers' 
                        is included in the header.

Function Process:
    - Constructs and prints a formatted header in the console.
    - The header is designed to visually segment different sections or stages of output, 
      making the console output easier to read and follow.

Return:
    None
"""

def print_header(title, subtitle=None, multiple_layers=False):

    lines = ["-----------------------------------------------------"]
    lines.append(title)

    if subtitle:
        lines.append(subtitle)

    if multiple_layers:
        lines.append("Multiple Layers")

    lines.append("-----------------------------------------------------")

    for line in lines:
        print(line)
