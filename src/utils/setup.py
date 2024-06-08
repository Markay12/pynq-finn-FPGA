"""
Setup for Testing
-------------------------

Description:
    This module provides a comprehensive setup for testing neural networks, particularly focusing on 
    noise injection and performance analysis in PyTorch models. It includes functions for data augmentation, 
    data loader creation, model setup, and various utilities to facilitate neural network testing in a 
    quantized environment. The emphasis is on ease of use and integration into existing testing workflows.

Functions:
    augment_data
    create_data_loaders
    extract_layer_names
    gen_rand_seed
    locate_hw_target
    load_dictionary
    model_criteria
    print_header
    setup_test

Usage:
    These utilities are intended to be imported and used in data processing or simulation scripts. Example of use:
    
        from setup import function1_name

        modified_data = function1_name(data)

Notes:
    - These functions are optimized for performance and accuracy. Most of these functions are tailored specifically to CIFAR100.
      However, this should be able to be changed and future functionality can be introduced to make this a parameter.
    
Author(s):
    Mark Ashinhust

Created on:
    17 January 2024

Last Modified:
    07 June 2024
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
import models

# Current script path
current_path = Path(__file__).parent

# Get parent directory
parent_directory = current_path

# Add parent to the Python path
sys.path.append(str(parent_directory))

def augment_data(crop_size, padding_size, mean_vector_train, std_vector_train, mean_vector_transform, std_vector_transform):
    """
    Define and apply data augmentation and transformations for CIFAR100 dataset.

    Args:
        crop_size (int): Size of the cropped image.
        padding_size (int): Size of padding around the image before cropping.
        mean_vector_train (list of float): Mean values for normalization in training data.
        std_vector_train (list of float): Standard deviation values for normalization in training data.
        mean_vector_transform (list of float): Mean values for normalization in validation data.
        std_vector_transform (list of float): Standard deviation values for normalization in validation data.

    Returns:
        train_set, val_set: Augmented and transformed training and validation datasets.
        train_transform, val_transform: Transformations applied to the datasets.
    """
    print_header("Defining Data Augmentation Transforms")

    train_transform = transforms.Compose([
        transforms.RandomCrop(crop_size, padding=padding_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vector_train, std=std_vector_train)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vector_transform, std=std_vector_transform)
    ])

    print_header("Applying Data Augmentation")

    train_set = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
    val_set = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=val_transform)

    return train_set, val_set, train_transform, val_transform

def create_data_loaders(train_set, val_set, train_transform, val_transform, batch_size):
    """
    Create data loaders for the training and validation datasets.

    Args:
        train_set (torchvision.datasets): Training dataset.
        val_set (torchvision.datasets): Validation dataset.
        train_transform (torchvision.transforms): Transformations applied to the training dataset.
        val_transform (torchvision.transforms): Transformations applied to the validation dataset.
        batch_size (int): Size of each batch of data.

    Returns:
        train_quantized_loader, val_quantized_loader: Data loaders for the training and validation datasets.
    """
    print_header("Creating Data Loaders")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)

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

def extract_layer_names(model):
    """
    Extract layer names from a neural network model.

    Args:
        model (torch.nn.Module): The neural network model.

    Returns:
        list: Array with the names of each individual layer of the model.
    """
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            cleaned_name = name.replace('.conv', '')
            layer_names.append(cleaned_name)
    return layer_names

def gen_rand_seed():
    """
    Generate a random seed based on the current date and time.

    Returns:
        None
    """
    time.sleep(1)
    print_header("Generating Random Seed")
    current_time = datetime.datetime.now()
    seed = int(current_time.timestamp())
    print(f"Random seed: {seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def locate_hw_target():
    """
    Identify the most suitable hardware target for running neural network models.

    Returns:
        torch.device: The selected hardware device (CUDA GPU or CPU).
    """
    print_header("Locating Targeted Device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Target device: " + str(device))
    return device

def load_dictionary(model, state_dict):
    """
    Load the provided state dictionary into the given model.

    Args:
        model (torch.nn.Module): The neural network model.
        state_dict (dict): The state dictionary containing model weights and biases.

    Returns:
        None
    """
    model.load_state_dict(state_dict)
    print_header("Printing Layer Shape Sizes")
    print("Convolutional Layer 1 Weight Shape: ", model.conv1.conv.weight.shape)
    print("Convolutional Layer 1 Bias: ", model.conv1.conv.bias)
    print("Fully Connected Layer 1 Weight Shape: ", model.fc1.weight.shape)

def model_criteria(device):
    """
    Initialize a neural network model, optimizer, scheduler, and criterion.

    Args:
        device (torch.device): The target device for model computation (CPU or GPU).

    Returns:
        model, state_dict, optimizer, scheduler, criterion, model_name: Initialized components for training and evaluation.
    """
    print_header("Model, Optimizer, and Criteria Initialized")

    model = models.CIFAR100CNN_5Blocks().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0009)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=11, factor=0.1, verbose=True)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 80
    best_test_accuracy = 0
    patience = 8
    no_improvement_counter = 0

    model_name = input("Name of the Model to Test: ")
    state_dict = torch.load(f'{os.getcwd()}/trained_models/{model_name}.pth', map_location=device)

    return model, state_dict, optimizer, scheduler, criterion, model_name

def print_header(title, subtitle=None, multiple_layers=False):
    """
    Print a formatted header in the console.

    Args:
        title (str): The main title for the header.
        subtitle (str, optional): An additional subtitle for more detailed headers.
        multiple_layers (bool, optional): Flag to indicate if the header is for multiple layers.

    Returns:
        None
    """
    lines = ["-----------------------------------------------------", title]
    if subtitle:
        lines.append(subtitle)
    if multiple_layers:
        lines.append("Multiple Layers")
    lines.append("-----------------------------------------------------")
    for line in lines:
        print(line)

def setup_test(crop_size, padding_size, mean_vector_train, std_vector_train, mean_vector_transform, std_vector_transform):
    """
    Main setup procedure for neural network testing.

    Args:
        crop_size, padding_size, mean_vector_train, std_vector_train, mean_vector_transform, std_vector_transform: Parameters for data augmentation and transformation.

    Returns:
        device, model, val_quantized_loader, model_name: Initialized components for further use.
    """
    device = locate_hw_target()
    train_set, val_set, train_transform, val_transform = augment_data(crop_size, padding_size, mean_vector_train, std_vector_train, mean_vector_transform, std_vector_transform)
    train_quantized_loader, val_quantized_loader = create_data_loaders(train_set, val_set, train_transform, val_transform, 256)
    model, state_dict, optimizer, scheduler, criterion, model_name = model_criteria(device)
    load_dictionary(model, state_dict)
    return device, model, val_quantized_loader, model_name
