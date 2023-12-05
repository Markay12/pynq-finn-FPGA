"""
Description: Each function is described individually and in full below.
Date: 24 August 2023

"""

#############################
#####      Imports      #####
#############################

## Randomness imports
from datetime import datetime
import random

## OS, csv Imports
import csv
import os

## Torch Import
import torch

## Matplotlib, numpy, deppcopy etc.. Imports
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np


def test(model, val_loader, device):
    """
    Evaluate a model's accuracy on a validation set.

    Parameters:
    - model : the PyTorch model to be tested.
    - val_loader : DataLoader object containing the validation data.
    - device : device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
    - accuracy (float) : the model's accuracy on the validation set.
    """

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def add_digital_noise(matrix, ber):
    """
    Add digital noise to a matrix.

    Parameters:
    - matrix (numpy array) : the matrix to which noise is to be added.
    - ber (float) : bit error rate or the probability of each element being flipped.

    Returns:
    - noisy_matrix (numpy array) : matrix with noise added.
    """

    random.seed(datetime.now().timestamp())
    mask = np.random.rand(*matrix.shape) < ber
    noise = np.random.uniform(-1, 1, size=matrix.shape)

    noisy_matrix = matrix + mask * noise

    return np.clip(noisy_matrix, -1, 1)

def add_digital_noise_to_model_brevitas(model, layer_names, ber, num_perturbations):
    """
    Add digital noise to the weight tensor of specific brevitas layers.

    Parameters:
    - model : original PyTorch model.
    - layer_names (list) : names of layers to which noise should be added.
    - ber (float) : bit error rate.
    - num_perturbations (int) : number of perturbations to perform.

    Returns:
    - modified_models (list) : list of models with added noise.
    """

    modified_models = []

    for _ in range(num_perturbations):
        modified_model = deepcopy(model)

        for layer_name in layer_names:
            # Determine layer type and retrieve weight tensor
            if layer_name.lower().startswith("c"):
                layer = getattr(modified_model, layer_name)
                weight_tensor = layer.conv.weight.cpu().detach()
            else:
                layer = getattr(modified_model, layer_name)
                weight_tensor = layer.weight.cpu().detach()

            with torch.no_grad():
                noisy_weight = add_digital_noise(weight_tensor.numpy(), ber)

                new_weight = torch.tensor(noisy_weight, dtype=torch.float)
                if layer_name.lower().startswith("c"):
                    layer.conv.weight = torch.nn.Parameter(new_weight)
                else:
                    layer.weight = torch.nn.Parameter(new_weight)

        modified_models.append(modified_model)

    return modified_models

def ber_noise_plot_brevitas(num_perturbations, layer_names, ber_vector, model, device, test_quantized_loader, model_name):
    """
    Plot the effect of BER noise on the test accuracy of a model.

    Parameters:
    - num_perturbations (int) : number of perturbations to perform.
    - layer_names (list) : names of layers to perturb.
    - ber_vector (list) : list of BER values to test.
    - model : the PyTorch model.
    - device : device to run the model on.
    - test_quantized_loader : DataLoader object for the test dataset.
    - model_name (str) : name of the model.

    Saves:
    - CSV file with raw accuracy data.
    - PNG plot of the test accuracy vs. BER value.
    """

    # Create directory for saving plots and data
    directory = os.path.join("noise_plots_brevitas", "ber_noise", model_name)
    os.makedirs(directory, exist_ok=True)

    all_test_accs = []

    # Set up CSV file to save raw data
    csv_file_path = os.path.join(directory, "raw_data.csv")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Layer", "BER Value", "Average Accuracy"])

        # Loop through each layer and BER value
        for layer in layer_names:
            test_accs = []

            for ber in ber_vector:
                noisy_models = add_digital_noise_to_model_brevitas(model, [layer], ber, num_perturbations)

                accuracies = [test(noisy_model.to(device), test_quantized_loader, device) for noisy_model in noisy_models]
                avg_accuracy = sum(accuracies) / len(accuracies)
                test_accs.append(avg_accuracy)

                writer.writerow([layer, ber, avg_accuracy])
                print(f"Layer: {layer}\tBER Value: {ber}\tAverage Accuracy: {avg_accuracy}")

            all_test_accs.append(test_accs)
            plt.plot(ber_vector, test_accs, label=f'{layer} Accuracy')

    avg_test_accs = [sum(x) / len(x) for x in zip(*all_test_accs)]

    plt.plot(ber_vector, avg_test_accs, label='Average', linewidth=3, linestyle='--', color="black")
    plt.xlabel('BER Value')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of BER Noise on Test Accuracy (Individual Layers and Average)')
    plt.legend(title='Layers')
    plt.savefig(os.path.join(directory, "individual_and_average.png"))
    plt.show()
