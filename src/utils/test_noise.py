"""
Test Utility
-------------------------

Description:
    The test utility defines how the models and neural networks will be tested. This function is in its own file 
    due to its frequent use throughout other utility files and the main script. The test function is a critical 
    component in the process of neural network analysis and allows for evaluating testing results.

Functions:
    test()

Notes:
    - This function is designed to be used with PyTorch models and DataLoader objects.
    
Author(s):
    Mark Ashinhust

Created on:
    08 December 2023

Last Modified:
    17 January 2024
"""

import torch

def test(model, test_loader, device):
    """
    Evaluate the accuracy of a neural network model on a test dataset.

    Args:
        model (torch.nn.Module): The neural network model to be tested.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
        float: The accuracy of the model on the test dataset, as a percentage.
    """
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total  # Calculate accuracy

    return accuracy
