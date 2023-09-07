
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

    # Generate random NxN matrix with values between 0 and 1
    N = weight.shape[0]
    M = weight.shape[1]
    
    weight_numpy = weight.cpu().numpy()
    
    if (N  > M):
        
        matrix = np.random.rand(N, N)
        L = N
        
    else:
        
        matrix = np.random.rand(M, M)
        L = M
    
            
    matrix_tensor = torch.tensor(matrix)
    
    # Compute 2D FFT
    fft_result = np.fft.fft2(matrix)

    # 1D Frequency Vector with N bins
    f = np.fft.fftfreq(L, d=1.0/L)
    f_x, f_y = np.meshgrid(f, f)
    f_x[0] = 1e-6
    f_y[0] = 1e-6

    # Create a 2D filter in frequency space that varies inversely with freq over f
    # Gamma controls the falloff rate
    filter_2D = 1/(np.sqrt(f_x**2 + f_y**2))**gamma

    # Mult the 2D elementwise by the filter
    filtered_fft = fft_result * filter_2D

    # 2D inverse FFT of the filtered result
    ifft_result = np.fft.ifft2(filtered_fft)
    ifft_result = np.real(ifft_result)

    # Set the threshold T equal the the max value in IFFT
    T = ifft_result.max()

    # Init empty bool mask with same dims as ifft
    mask = np.zeros_like(ifft_result, dtype=bool)

    decrement_step = 0.01

    # Repeat until frac of nonzero values in the mask is greater than or equal to P
    timeout = 20   # [seconds]
    timeout_start = time.time()
    while time.time() < (timeout_start + timeout):
        mask = ifft_result > T

        current_fraction = np.count_nonzero(mask) / (N * N)

        if current_fraction >= P:
            break

        T -= decrement_step

    print("Took"+ str(time.time()-timeout_start)+ "to converge")
    # Return tensor with the same shape as the input tensor
    # mask = np.tile(mask, (weight_shape[0], weight_shape[1], 1, 1))
    
    if (N > M):
        
        mask = mask[:,:M]
        
    else:
        
        mask = mask[:N,:]
        
    mask_final = np.zeros_like(weight_numpy)
    #print(mask_final.shape)
    #print(mask.shape)
    
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=weight.device)
    mask_final = torch.zeros_like(weight)

    if len(weight.shape) == 4:
        # Create a temporary tensor that repeats the mask tensor along the 3rd dimension
        temp_mask = mask_tensor.unsqueeze(2).repeat(1, 1, weight.shape[2])

        # Copy the temporary mask tensor along the 4th dimension (axis=3) of the mask_final tensor
        for i in range(mask_final.shape[3]):
            mask_final[:, :, :, i] = temp_mask

    elif len(weight.shape) == 2:
        mask_final = torch.tensor(mask, device = weight.device)
        
    mask_logical = torch.zeros(weight.shape, dtype=torch.bool,device=weight.device)
    mask_logical = (mask_final==1)
    mask_numeric = mask_final
    
    #mask tensor is 2D and float type
    #mask numeric/final is 4D and float like
    #mask logical is 4D and bool type
    return mask_numeric, mask_logical


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


