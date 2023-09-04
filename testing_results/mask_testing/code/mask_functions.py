
## Imports
from datetime import datetime
import random
import os
import csv
import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import time

"""
The function sets the model to evaluation mode, disables gradient computation, 
and iterates through the test data loader, passing each batch of images through the 
model to get predicted labels. It then compares the predicted labels to the true labels 
and calculates the percentage of correct predictions (accuracy) over the entire test 
dataset. Finally, the function returns the accuracy value as a float.
"""

def test(model, val_loader, device):
    
    # testing phase
    model.eval()

    correct = 0
    total = 0
    
    with torch.no_grad():

        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            
    accuracy = 100 * correct / total
    return accuracy

###################################
# Mask Section
###################################

## Noise Mask Section

"""
The random_clust_mask function generates a random clustered mask of boolean values with a given size N and target density P. 
It first generates a random matrix of values between 0 and 1, performs a two-dimensional Fast Fourier Transform on this matrix, 
and creates a 2D filter in frequency space that varies inversely with frequency over a frequency vector f. The filter's falloff 
rate is controlled by a parameter gamma. The filtered result is then inverse transformed back to the spatial domain, and a 
threshold T is set to the maximum value in the result. Finally, a boolean mask is initialized with the same dimensions as the 
result, and the threshold T is decreased iteratively until the fraction of nonzero values in the mask is greater than or equal 
to the target density P. The function returns the boolean mask as a PyTorch tensor.
"""

def random_clust_mask(weight, P, gamma):
    """
    Generates a random clustered mask based on the input weight tensor.
    
    Args:
        weight (torch.Tensor): Input weight tensor.
        P (float): The target fraction of non-zero values in the mask.
        gamma (float): Controls the falloff rate of the 2D filter in frequency space.

    Returns:
        torch.Tensor: A mask tensor of the same shape as the input weight tensor.
    """
    
    # Seed the RNG
    random.seed(datetime.now().timestamp())
    
    weight_shape = weight.shape
    
    # Check the shape of the weight tensor
    if len(weight_shape) == 4:
        N, M = weight.shape[2], weight.shape[3]
    elif len(weight_shape) == 2:
        N, M = weight.shape[0], weight.shape[1]
    else:
        raise ValueError("Unexpected weight dimensions")
    
    # Create an initial matrix based on the larger of N and M
    if N > M:
        matrix = torch.rand((N, N), device=weight.device)
        L = N
    else:
        matrix = torch.rand((M, M), device=weight.device)
        L = M
    
    # Compute the 2D FFT of the matrix
    fft_result = torch.fft.fft2(matrix)
    
    # Generate a 2D frequency filter
    f = torch.fft.fftfreq(L, d=1.0/L)
    f_x, f_y = torch.meshgrid(f, f)
    f_x[0], f_y[0] = 1e-6, 1e-6
    filter_2D = 1 / (torch.sqrt(f_x**2 + f_y**2))**gamma
    
    # Apply the filter and perform inverse FFT
    ifft_result = torch.fft.ifft2(fft_result * filter_2D).real
    
    # Binary search approach to determine threshold T
    low, high = ifft_result.min().item(), ifft_result.max().item()
    while high - low > 1e-5:
        T = (low + high) / 2
        mask = (ifft_result > T).float()
        current_fraction = mask.sum().item() / (L * L)
        
        if current_fraction > P:
            low = T
        else:
            high = T

    converged_T = T

    # Adjust mask dimensions to match the input weight tensor
    if N > M:
        mask = mask[:, :M]
    else:
        mask = mask[:N, :]
    
    if len(weight_shape) == 4:
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask_final = mask.expand(weight_shape[0], weight_shape[1], -1, -1)
    elif len(weight_shape) == 2:
        mask_final = mask
    
    mask_logical = (mask_final == 1).float()
    
    return mask_final, mask_logical



"""
This function takes a brevitas layer, a mask generated by the random_clust_mask function,
and two parameters P and gamma. It generates a random clustered mask using the random_clust_mask
function, and then sets the weights at the locations in the mask to zero, effectively sparsifying
the layer's weights.
"""

def get_weight_tensor_from_layer(layer, layer_name):
    """Retrieve the weight tensor from a layer based on its type."""
    if layer_name.lower().startswith("c"):
        
        # Access the convolutional weights
        return layer.conv.weight.cpu().clone().detach()
    else:
        return layer.weight.cpu().clone().detach()
    
def return_noisy_matrix(independent_type, shape, sigma, device):
    """
    Returns a matrix of noise.

    Args:
        independent_type (bool): Determines if the noise should be independent or proportional.
        shape (tuple): Shape of the matrix to return.
        sigma (float): Standard deviation of the Gaussian noise.
        device (torch.device): Device for the tensor.

    Returns:
        torch.Tensor: A tensor of the same shape filled with Gaussian noise.
    """
    if independent_type:
        # Independent Gaussian noise
        noisy_matrix = torch.randn(shape, device=device) * sigma
    else:
        # Proportional Gaussian noise
        # If you mean "proportional" as in proportional to some other matrix values, you'd multiply the Gaussian noise by that matrix here.
        # As of now, I'm assuming it's independent if this is not the case.
        noisy_matrix = torch.randn(shape, device=device) * sigma
        
    return noisy_matrix


def apply_mask_and_noise_to_weights(weight_tensor, mask_numeric, mask_logical, independent_type, sigma, device):
    """Apply mask and noise to the weights based on the type."""

    weight_tensor = weight_tensor.to(device)
    mask_numeric = mask_numeric.to(device)
    mask_logical = mask_logical.to(device)
    
    if weight_tensor.device != mask_numeric.device:
        mask_numeric = mask_numeric.to(weight_tensor.device)
        mask_logical = mask_logical.to(weight_tensor.device)

    noisy_matrix = return_noisy_matrix(independent_type, weight_tensor.shape, sigma, device)

    if len(weight_tensor.shape) == 4:
        return apply_mask_and_noise_to_conv_weights(weight_tensor, mask_numeric, mask_logical, independent_type, noisy_matrix)
    elif len(weight_tensor.shape) == 2:
        return apply_mask_and_noise_to_linear_weights(weight_tensor, mask_numeric, mask_logical, independent_type, noisy_matrix)
    else:
        raise ValueError("Unexpected weight dimensions")

def apply_mask_and_noise_to_conv_weights(weight_tensor, mask_numeric, mask_logical, independent_type, noisy_matrix):

    if independent_type:
        weight_tensor += mask_numeric * noisy_matrix
    else:
        noisy_matrix[torch.logical_not(mask_logical)] = 1
        weight_tensor *= noisy_matrix
        

    return weight_tensor

def apply_mask_and_noise_to_linear_weights(weight_tensor, mask_numeric, mask_logical, independent_type, noisy_matrix):

    if independent_type:
        weight_tensor += mask_numeric * noisy_matrix
    else:
        noisy_matrix[torch.logical_not(mask_logical)] = 1
        weight_tensor *= noisy_matrix

    return weight_tensor

from torch.nn import Conv2d, Linear


def add_mask_to_model_brevitas(model, device, layer_names, p, gamma, num_perturbations, sigma, independent_type, print_weights=False):
    """Modify model weights by applying masks and noise, returning a list of perturbed models."""
    modified_models = []

    for _ in range(num_perturbations):
        modified_model = deepcopy(model)

        for layer_name in layer_names:
            layer = getattr(modified_model, layer_name)
            weight_tensor = get_weight_tensor_from_layer(layer, layer_name)

            print(weight_tensor.shape)

            # Generate mask with correct shape
            mask_numeric, mask_logical = random_clust_mask(weight_tensor, p, gamma)

            # For debugging: Print weights before the mask application
            if print_weights:
                print("Weights before masking:")
                print(weight_tensor)

            # Apply mask and noise to the weight tensor
            if layer_name.lower().startswith("c"):
                # Logic specific to convolutional layers (if any) can go here
                weight_tensor = apply_mask_and_noise_to_weights(weight_tensor, mask_numeric, mask_logical, independent_type, sigma, device)
            elif isinstance(layer, torch.nn.Linear):
                # Logic specific to fully connected layers (if any) can go here
                weight_tensor = apply_mask_and_noise_to_weights(weight_tensor, mask_numeric, mask_logical, independent_type, sigma, device)

            # For debugging: Print weights after the mask application
            if print_weights:
                print("Weights after masking:")
                print(weight_tensor)

            # Adjust this assignment based on layer type
            if layer_name.lower().startswith("c"):
                layer.conv.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
            else:
                layer.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                
        modified_models.append(modified_model)

    return modified_models



"""
The function mask_noise_plots_brevitas() is a Python function that is used to visualize the effect of different perturbations on the accuracy of a neural network. 
It generates several visualizations, including a heatmap and a scatter plot.
The function takes in the following parameters:
    num_perturbations: an integer indicating the number of perturbations to apply to the neural network
    layer_names: a list of strings indicating the names of the layers in the neural network
    p_vals: a list of floats indicating the p values to use for the mask perturbation
    gamma_vals: a list of floats indicating the gamma values to use for the mask perturbation
    model: the neural network model to test
    device: the device to use for testing the model
The function begins by creating a directory for saving the output plots. 
It then initializes an empty list called all_test_accs to store the test accuracies for each layer.
For each layer in layer_names, the function initializes an empty list called test_accs to store 
the test accuracies for each mask perturbation. The function then iterates over each p and gamma value 
and adds noise to the model for the defined layer only. It then tests the accuracy of each noisy model 
and appends the result to the accuracies list. The function then calculates the average accuracy 
and prints the result.
The test accuracies for the current layer are then stored in the all_test_accs list.
The function then creates a heatmap for each layer and saves it to disk.
It also computes the average test accuracy across all layers for each p and gamma value and creates a
heatmap for the average test accuracy. Both the individual and average heatmaps have a color bar 
indicating the test accuracy.
Finally, the function creates a scatter plot showing the test accuracies for each layer at each p 
and gamma value. The plot has layer_names on the x-axis, p_vals on the y-axis, and gamma_vals on the z-axis. 
The points in the plot are colored based on the corresponding test accuracy, with a color bar indicating 
the mapping between color and test accuracy.
"""

def mask_noise_plots_brevitas(num_perturbations, layer_names, p_values, gamma_values, model, device, sigma_values, independent_type, test_quantized_loader, model_name):

    # Check if output directory exists, if not, create it
    if independent_type:
        if not os.path.exists(f"noise_plots_brevitas/mask/{model_name}/line_plot/independent"):
            os.makedirs(f"noise_plots_brevitas/mask/{model_name}/line_plot/independent")
            
        output_dir = f"noise_plots_brevitas/mask/{model_name}/line_plot/independent"

        if not os.path.exists(f"noise_plots_brevitas/mask/{model_name}/data"):
            os.makedirs(f"noise_plots_brevitas/mask/{model_name}/data")

        csv_file_path = os.path.join(f"noise_plots_brevitas/mask/{model_name}/data", "raw_data_independent.csv")

    else:
        if not os.path.exists(f"noise_plots_brevitas/mask/{model_name}/line_plot/proportional"):
            os.makedirs(f"noise_plots_brevitas/mask/{model_name}/line_plot/proportional")
        output_dir = f"noise_plots_brevitas/mask/{model_name}/line_plot/proportional"

        if not os.path.exists(f"noise_plots_brevitas/mask/{model_name}/data"):
            os.makedirs(f"noise_plots_brevitas/mask/{model_name}/data")

        csv_file_path = os.path.join(f"noise_plots_brevitas/mask/{model_name}/data", "raw_data_proportional.csv")


    # Create a CSV file to store the raw data

    random.seed(datetime.now().timestamp())

    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Layer", "P Value", "Gamma Value", "Sigma Value", "Average Accuracy"])

        for layer in layer_names:
            # Initialize a 3D array to store average accuracies for each combination of p, gamma, and sigma values
            avg_accuracies = np.zeros((len(p_values), len(gamma_values), len(sigma_values)))

            for p in range(len(p_values)):
                for g in range(len(gamma_values)):
                    for s in range(len(sigma_values)):
                        noisy_models = add_mask_to_model_brevitas(model, device, [layer], p_values[p], gamma_values[g], num_perturbations, sigma_values[s], independent_type)

                        accuracies = []

                        for noisy_model in noisy_models:
                            noisy_model.to(device)
                            accuracies.append(test(noisy_model, test_quantized_loader, device))

                        avg_accuracy = sum(accuracies) / len(accuracies)
                        avg_accuracies[p, g, s] = avg_accuracy

                        # Write the raw data to the CSV file
                        writer.writerow([layer, p_values[p], gamma_values[g], sigma_values[s], avg_accuracy])

                        print("Layer: {}\tP Value: {}\t Gamma Value: {}\t Sigma Value: {}\t Average Accuracy: {}%".format(layer, p_values[p], gamma_values[g], sigma_values[s], avg_accuracy))

            for p in range(len(p_values)):
                plt.figure()

                # Use different colors for each gamma value
                color_map = plt.get_cmap('viridis')
                colors = color_map(np.linspace(0, 1, len(gamma_values)))

                for g in range(len(gamma_values)):
                    avg_accuracy_gamma = avg_accuracies[p, g, :]
                    gamma_truncated = round(gamma_values[g], 3)
                    plt.plot(sigma_values, avg_accuracy_gamma, color=colors[g], label=f'Gamma: {gamma_truncated}')

                plt.xlabel('Sigma Value')
                plt.ylabel('Average Accuracy')
                plt.title(f'Line Plot for P Value: {p_values[p]}')
                plt.legend(title='Gamma Values')
                plt.savefig(os.path.join(output_dir, f'line_plot_p_{p_values[p]}_{layer}.png'))
                plt.clf()


def mask_noise_plots_brevitas_multiple_layers(num_perturbations, layer_names_list, p_values, gamma_values, model, device, sigma_values, independent_type, test_quantized_loader, model_name):

    # Check if output directory exists, if not, create it
    if independent_type:
        if not os.path.exists(f"noise_plots_brevitas/mask/{model_name}/line_plot/independent"):
            os.makedirs(f"noise_plots_brevitas/mask/{model_name}/line_plot/independent")
            
        output_dir = f"noise_plots_brevitas/mask/{model_name}/line_plot/independent"

        if not os.path.exists(f"noise_plots_brevitas/mask/{model_name}/data/all_layers"):
            os.makedirs(f"noise_plots_brevitas/mask/{model_name}/data/all_layers")

        csv_file_path = os.path.join(f"noise_plots_brevitas/mask/{model_name}/data/all_layers", "raw_data_independent_all.csv")

    else:
        if not os.path.exists(f"noise_plots_brevitas/mask/{model_name}/line_plot/proportional"):
            os.makedirs(f"noise_plots_brevitas/mask/{model_name}/line_plot/proportional")
        output_dir = f"noise_plots_brevitas/mask/{model_name}/line_plot/proportional"

        if not os.path.exists(f"noise_plots_brevitas/mask/{model_name}/data/all_layers"):
            os.makedirs(f"noise_plots_brevitas/mask/{model_name}/data/all_layers")

        csv_file_path = os.path.join(f"noise_plots_brevitas/mask/{model_name}/data/all_layers", "raw_data_proportional_all.csv")

    # Create a CSV file to store the raw data
    csv_file_path = os.path.join(output_dir, "raw_data_all_layers.csv")

    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Layers", "P Value", "Gamma Value", "Sigma Value", "Average Accuracy"])

        for layer_names in layer_names_list:
            layer_combo_name = '_'.join(layer_names)
            avg_accuracies = np.zeros((len(p_values), len(gamma_values), len(sigma_values)))

            for p in range(len(p_values)):
                for g in range(len(gamma_values)):
                    for s in range(len(sigma_values)):
                        noisy_models = add_mask_to_model_brevitas(model, device, layer_names, p_values[p], gamma_values[g], num_perturbations, sigma_values[s], independent_type)

                        accuracies = []

                        for noisy_model in noisy_models:
                            noisy_model.to(device)
                            accuracies.append(test(noisy_model, test_quantized_loader, device))

                        avg_accuracy = sum(accuracies) / len(accuracies)
                        avg_accuracies[p, g, s] = avg_accuracy

                        # Write the raw data to the CSV file
                        writer.writerow([layer_combo_name, p_values[p], gamma_values[g], sigma_values[s], avg_accuracy])

                        print("Layers: {}\tP Value: {}\t Gamma Value: {}\t Sigma Value: {}\t Average Accuracy: {}%".format(layer_combo_name, p_values[p], gamma_values[g], sigma_values[s], avg_accuracy))

            for p in range(len(p_values)):
                plt.figure()

                color_map = plt.get_cmap('viridis')
                colors = color_map(np.linspace(0, 1, len(gamma_values)))

                for g in range(len(gamma_values)):
                    avg_accuracy_gamma = avg_accuracies[p, g, :]
                    gamma_truncated = round(gamma_values[g], 3)
                    plt.plot(sigma_values, avg_accuracy_gamma, color=colors[g], label=f'Gamma: {gamma_truncated}')

                plt.xlabel('Sigma Value')
                plt.ylabel('Average Accuracy')
                plt.title(f'Line Plot for P Value: {p_values[p]}')
                plt.legend(title='Gamma Values')
                plt.savefig(os.path.join(output_dir, f'line_plot_p_{p_values[p]}_{layer_combo_name}.png'))
                plt.clf()


