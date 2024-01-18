"""
Setup for Testing
-------------------------

Description:
    Description here.

Functions:
    Functions_go_here()

Usage:
    These utilities are intended to be imported and used in data processing or
    simulation scripts. Example of use:
    
        from noise_injection import function1_name

        modified_data = function1_name(data)

Notes:
    - These functions are optimized for performance and accuracy.
    
Author(s):
    Mark Ashinhust

Created on:
    17 January 2024

Last Modified:
    17 January 2024

"""

import torch
import torchvision
from torchvision import transforms

def locate_hw_target():

    # print statement for debug
    print("-----------------------------------------------------")
    print("-------------- Locating Targeted Device -------------")
    print("-----------------------------------------------------")

    # Locate the device that can be used for noise analysis and testing
    # Target this device 
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    print( "Target device: " + str (device ) )

def augment_data( crop_size, padding_size, mean_vector_train, std_vector_train, mean_vector_transform, std_vector_transform ):

    # print statement for debug
    print("-----------------------------------------------------")
    print("------- Defining Data Augmentation Transforms -------")
    print("-----------------------------------------------------")

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

    print("-----------------------------------------------------")
    print("------------ Applying Data Augmentation -------------")
    print("-----------------------------------------------------")

    # Apply data augmentation to the training dataset
    train_set = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)

    # Use the validation transform for the validation dataset
    val_set = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=val_transform)


def create_data_loaders():

    # print for debugging
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