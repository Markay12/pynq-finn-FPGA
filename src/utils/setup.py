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

    # return whether using CUDA or CPU
    return device

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

    return train_set, val_set, train_transform, val_transform

def create_data_loaders( train_set, val_set, train_transform, val_transform, batch_size ):

    # print statement for debug
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


def model_criteria( device ):

    # print for debug
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

    return model, state_dict, optimizer, scheduler, criterion, model_name

def load_dictionary( model, state_dict ):

    model, state_dict = model_criteria()

    # Load the state dictionary into the model
    model.load_state_dict( state_dict )

    print("-----------------------------------------------------")
    print("------------- Printing Layer Shape Sizes ------------")
    print("-----------------------------------------------------")

    # Test shapes
    print( "Convolutional Layer 1 Weight Shape: ",   model.conv1.conv.weight.shape )
    print( "Convolutional Layer 1 Bias: ",           model.conv1.conv.bias )
    print( "Fully Connected Layer 1 Weight Shape: ", model.fc1.weight.shape )


def setup_test( crop_size, padding_size, mean_vector_train, std_vector_train, mean_vector_transform, std_vector_transform ):

    device = locate_hw_target()
    
    train_set, val_set, train_transform, val_transform = augment_data(crop_size, padding_size, mean_vector_train, std_vector_train, mean_vector_transform, std_vector_transform)
    train_quantized_loader, val_quantized_loader = create_data_loaders(train_set, val_set, train_transform, val_transform)
    model, state_dict, optimizer, scheduler, criterion, model_name = model_criteria(device)
    
    load_dictionary( model, state_dict )

    return device, model, val_quantized_loader, model_name

