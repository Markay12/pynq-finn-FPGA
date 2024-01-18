"""
Test Utility
-------------------------

Description:
    The test utility file here defines how the models and neural networks will be tested. This function is in its own file due
    to it being used in countless other situations throughout the other utility files and within the main script.
    This test function is one of the last used functions in the process of this neural network analysis and is what 
    allows for testing results.

Functions:
    test()

Notes:
    
Author(s):
    Mark Ashinhust

Created on:
    08 December 2023

Last Modified:
    17 January 2024

"""
## IMPORT STATEMENTS
import torch

"""
The function sets the model to evaluation mode, disables gradient computation, 
and iterates through the test data loader, passing each batch of images through the 
model to get predicted labels. It then compares the predicted labels to the true labels 
and calculates the percentage of correct predictions (accuracy) over the entire test 
dataset. Finally, the function returns the accuracy value as a float.
"""

def test(model, test_loader, device):

    # testing phase
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    
    return accuracy
