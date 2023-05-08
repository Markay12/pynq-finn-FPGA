## Imports
import numpy as np
import torch
import random
from copy import deepcopy
import os
import pandas as pd
import matplotlib.pyplot as plt

"""
This file contains functions for adding noise to different models. The functions included are:

    random_clust_mask: Generates a random clustered mask for adding to a layer's weights
    add_mask_to_weights: Applies a mask generated by random_clust_mask to a layer's weights
    add_digital_noise: Applies digital noise to a matrix
    add_noise: Adds Gaussian noise to a matrix
    add_noise_to_model: Adds Gaussian noise to specified layers in a model and returns modified models

These functions can be imported and used in other Python scripts as needed.
"""


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

def random_clust_mask(weight_shape, P, gamma):

    # Generate random NxN matrix with values between 0 and 1
    N = weight_shape[-1]
    matrix = np.random.rand(N, N)

    # Compute 2D FFT
    fft_result = np.fft.fft2(matrix)

    # 1D Frequency Vector with N bins
    f = np.fft.fftfreq(N, d=1.0/weight_shape[-1])
    f_x, f_y = np.meshgrid(f, f)
    f_x[0, 0] = 1e-6
    f_y[0, 0] = 1e-6

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
    while True:
        mask = ifft_result > T

        current_fraction = np.count_nonzero(mask) / (N * N)

        if current_fraction >= P:
            break

        T -= decrement_step

    # Return tensor with the same shape as the input tensor
    mask = np.tile(mask, (weight_shape[0], weight_shape[1], 1, 1))
    return torch.tensor(mask, dtype=torch.float)



"""
This function takes a brevitas layer, a mask generated by the random_clust_mask function,
and two parameters P and gamma. It generates a random clustered mask using the random_clust_mask
function, and then sets the weights at the locations in the mask to zero, effectively sparsifying
the layer's weights.
"""

def add_mask_to_model_brevitas(model, layer_names, p, gamma, num_perturbations):

    modified_models = []

    for _ in range(num_perturbations):

        modified_model = deepcopy(model)

        for layer_name in layer_names:

            layer = getattr(modified_model, layer_name)

            with torch.no_grad():

                # get weights and biases of the tensors
                weight = layer.weight.cpu().detach().numpy()

                # generate mask with correct shape
                mask = random_clust_mask(weight.shape, p, gamma)

                # convert weight tensor to PyTorch tensor
                weight_tensor = torch.tensor(weight, dtype=torch.float)

                # apply mask to the whole weight tensor
                mask_tensor = torch.tensor(mask, dtype=torch.float).clone().detach()  # Change this line
                weight_tensor *= mask_tensor  # Change this line

                # create new weight parameter and assign to layer
                noised_weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                layer.weight = noised_weight

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

def mask_noise_plots_brevitas(num_perturbations, layer_names, p_values, gamma_values, model, device, test_quantized_loader):
    if not os.path.exists("noise_plots_brevitas/mask/"):
        os.makedirs("noise_plots_brevitas/mask")
        
    plt.style.use('default')
    
    # Create a list to store the test accuracies for all layers
    all_test_accs = []
    
    # Loop over each layer
    for layer in layer_names:
        # Initialize a list to store the test accuracies for this layer
        test_accs = []
        
        # Iterate over p_values and gamma_values
        for p in p_values:
            for gamma in gamma_values:
                
                # Add noise to the model for the defined layer only
                noisy_models = add_mask_to_model_brevitas(
                    model, [layer], p, gamma, num_perturbations)
                accuracies = []
                
                # Test the accuracy of each noisy model and append the result to the accuracies list
                for noisy_model in noisy_models:
                    # Move the model back to the target device
                    noisy_model.to(device)
                    accuracies.append(test(noisy_model, test_quantized_loader, device))
                
                # Calculate the average accuracy and print the result
                avg_accuracy = sum(accuracies) / len(accuracies)
                test_accs.append(avg_accuracy)
                print("Layer: {}, p: {}, gamma: {}, Average Accuracy: {}%".format(
                    layer, p, gamma, avg_accuracy))
        
        # Store the test accuracies for this layer in the all_test_accs list
        all_test_accs.append(test_accs)
    
    # Define the grid for p_values and gamma_values
    p_grid, gamma_grid = np.meshgrid(p_values, gamma_values)
    
    for i, layer in enumerate(layer_names):
        # Reshape test_accs list into a matrix
        test_accs_matrix = np.reshape(all_test_accs[i], (len(p_values), len(gamma_values)))
    
        # Create heatmap for the current layer
        plt.figure()
        plt.imshow(test_accs_matrix, cmap='viridis', origin='lower',
                   extent=(p_values[0], p_values[-1], gamma_values[0], gamma_values[-1]),
                   aspect='auto')
        plt.colorbar(label='Test Accuracy')
        plt.xlabel('P value')
        plt.ylabel('Gamma value')
        plt.title(f'Effect of Mask on Test Accuracy for {layer}')
        plt.savefig(f"noise_plots_brevitas/mask/{layer}.png")
        plt.clf()
    
    # Compute the average test accuracy across all layers for each p and gamma value
    avg_test_accs = np.mean(all_test_accs, axis=0)
    avg_test_accs_matrix = np.reshape(avg_test_accs, (len(p_values), len(gamma_values)))
    
    # Create heatmap for the average test accuracy
    plt.figure()
    plt.imshow(avg_test_accs_matrix, cmap='viridis', origin='lower',
               extent=(p_values[0], p_values[-1], gamma_values[0], gamma_values[-1]),
               aspect='auto')
    plt.colorbar(label='Test Accuracy')
    plt.xlabel('P value')
    plt.ylabel('Gamma Value')
    plt.title('Effect of Mask on Test Accuracy (Average)')
    plt.savefig("noise_plots_brevitas/mask/average.png")
    plt.show()
    plt.clf()
    
    
    # Begin Scatter Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Define the grid for p_values and gamma_values
    p_grid, gamma_grid = np.meshgrid(p_values, gamma_values)
    
    for i, layer in enumerate(layer_names):
        # Reshape test_accs list into a matrix
        test_accs_matrix = np.reshape(all_test_accs[i], (len(p_values), len(gamma_values)))
        
        # Flatten the matrices
        pvals_mesh = p_grid.flatten()
        gammavals_mesh = gamma_grid.flatten()
        layer_indices_mesh = np.full_like(pvals_mesh, i)  # create an array of layer indices
        
        # Scatter plot with layer indices, pvals, gammavals, and corresponding test accuracies
        c_array = np.array(test_accs_matrix).flatten()
        scatter = ax.scatter(layer_indices_mesh, pvals_mesh, gammavals_mesh, c=c_array, cmap='viridis', marker='o', s=60)
    
    # Customize the plot
    ax.set_xlabel('Layer index')
    ax.set_ylabel('P value')
    ax.set_zlabel('Gamma value')
    ax.set_xticks(np.arange(len(layer_names)))
    ax.set_xticklabels(layer_names)
    
    # Add color bar to the plot
    cbar = fig.colorbar(scatter, ax=ax, label='Test Accuracy')
    
    # Save and show the plot
    plt.savefig("noise_plots_brevitas/mask/scatter_plot.png")
    plt.show()

    # Create a plot with Y-axis: Accuracy on test following perturbation
    # X-axis: P
    # Multiple line series where each line series is given the gamma/clustering param
    plt.figure()
    for i, gamma in enumerate(gamma_values):
        # Extract the average test accuracy for each p value at the current gamma value
        avg_test_accs_gamma = avg_test_accs_matrix[:, i]

        # Plot the line series for the current gamma value
        plt.plot(p_values, avg_test_accs_gamma, label=f"Gamma = {gamma}")

    # Customize the plot
    plt.xlabel('P value')
    plt.ylabel('Accuracy on test following perturbation')
    plt.title('Effect of Mask on Test Accuracy (Gamma-wise)')
    plt.legend()

    # Save and show the plot
    plt.savefig("noise_plots_brevitas/mask/gamma_wise_plot.png")
    plt.show()



## Digital Noise Section

"""
This function adds digital noise to a matrix. It takes in two arguments, matrix, 
which is the matrix to which noise is to be added and ber, which is the bit error rate 
or the probability of each element of the matrix being flipped.

First, the function generates a boolean mask with the same shape as the matrix where each
element is True with probability ber. Then it generates a matrix of random values between 
-1 and 1 with the same shape as the input matrix. The noise is added to the input matrix 
only where the mask is True. Finally, the function clips the noisy matrix to the range [-1, 1] 
to ensure that all elements are within this range and returns the noisy matrix.
"""

def add_digital_noise(matrix, ber):
    # Generate a boolean mask with the same shape as the matrix where each element is True with probability `ber`
    mask = np.random.rand(*matrix.shape) < ber
    
    # Generate a matrix of random values between -1 and 1 with the same shape as the input matrix
    noise = np.random.uniform(-1, 1, size=matrix.shape)
    
    # Add the noise to the input matrix only where the mask is True
    noisy_matrix = matrix + mask * noise
    
    # Clip the noisy matrix to the range [-1, 1] to ensure that all elements are within this range
    return np.clip(noisy_matrix, -1, 1)

"""
The function add_digital_noise_to_brevitas_layer adds digital noise to the weight tensor of a brevitas layer. 
It first gets the weight tensor of the layer and converts it to a numpy array. Then, it applies digital noise 
to the weight tensor using the add_digital_noise function. The noisy weight tensor is converted back to a 
PyTorch tensor and is used to update the layer's weight tensor.
"""


def add_digital_noise_to_model_brevitas(model, layer_names, ber, num_perturbations):
    
    modified_models = []
    
    for _ in range(num_perturbations):
        
        modified_model = deepcopy(model)
        
        for layer_name in layer_names:
            
            layer = getattr(modified_model, layer_name)
            
            with torch.no_grad():
                
                weight = layer.weight.cpu().detach().numpy()
                noisy_weight = add_digital_noise(weight, ber)
                
                
                layer.weight = torch.nn.Parameter(torch.tensor(noisy_weight, dtype=torch.float))
                
        modified_models.append(modified_model)
        
    return modified_models


"""
This function plots the effect of bit error rate (BER) noise on the test accuracy of a given model. 
It takes in the number of perturbations, the layer names, a vector of BER values, the model and the 
device as inputs.

It first creates a directory to save the generated plots if it doesn't already exist. 
It then initializes an empty list to store all the test accuracies.

The function then loops over each layer in the provided layer names list and BER values in the BER vector. 
For each combination of layer and BER value, it adds digital noise to the weights of the layer for a certain 
number of perturbations and tests the accuracy of the noisy model on the test dataset. It then calculates 
the average accuracy across all the noisy models generated and appends it to the test accuracies list. 
It also generates a plot of the accuracy at different perturbation levels for each layer and saves it in 
the previously created directory.

After all the layers have been looped through, the function calculates the average test accuracy for each 
BER value by taking the mean of the accuracies across all layers. It then generates a plot of the average 
test accuracy at different BER values and saves it in the directory.
"""


def ber_noise_plot_brevitas(num_perturbations, layer_names, ber_vector, model, device, test_quantized_loader):
    
    if not os.path.exists("noise_plots_brevitas/ber_noise/"):
        os.makedirs("noise_plots_brevitas/ber_noise/")
    
    plt.style.use('default')
    
    all_test_accs = []

    for layer in layer_names:
        test_accs = []
        
        for ber in ber_vector:
            noisy_models = add_digital_noise_to_model_brevitas(model, [layer], ber, num_perturbations)
            
            accuracies = []
            
            for noisy_model in noisy_models:
                
                noisy_model.to(device)
                accuracies.append(test(noisy_model, test_quantized_loader, device))
                
            avg_accuracy = sum(accuracies) / len(accuracies)
            
            test_accs.append(avg_accuracy)
            
            print("BER Value: {}\tAverage Accuracy: {}".format(ber, avg_accuracy))
            
        all_test_accs.append(test_accs)
        
        plt.plot(ber_vector, test_accs,
                 label='{} Accuracy at Different Perturbation Levels'.format(layer))
        
        plt.xlabel('Standard Deviation')
        plt.ylabel('Test Accuracy')
        plt.title('Effect of Noise on Test Accuracy')
        plt.legend()
        plt.savefig("noise_plots_brevitas/ber_noise/{}.png".format(layer))
        plt.clf()
        print('Done with Plot {}'.format(layer))
        
    avg_test_accs = [sum(x) / len(x) for x in zip(*all_test_accs)]
    
    plt.plot(ber_vector, avg_test_accs, label='Average',
             linewidth=3, linestyle='--', color="black")
    
    plt.xlabel('BER Value')
    
    plt.ylabel('Test Accuracy')
    
    plt.title('Effect of BER Noise on Test Accuracy (Average)')
    
    plt.legend()
    plt.savefig("noise_plots_brevitas/ber_noise/average.png")
    plt.clf()
    



## Gaussian Noise Section

"""
This function adds Gaussian noise to a given matrix. The sigma parameter controls
the standard deviation of the noise distribution. The function returns the original
matrix with added noise.
"""

def add_gaussian_noise_independent(matrix, sigma):
    noised_matrix = matrix + np.random.normal(0, scale=sigma)
    return noised_matrix

def add_gaussian_noise_proportional(matrix, sigma):
    noised_matrix = matrix * np.random.normal(1, scale=sigma)
    return noised_matrix


def return_noisy_matrix(independent, dim_matrix, sigma):
    
    if (independent):
        noised_matrix = torch.randn(dim_matrix) * sigma
    else:
        noised_matrix = torch.randn(dim_matrix) * sigma + 1

    return noised_matrix


"""
The add_gaussian_noise_to_model_brevitas function takes a Brevitas model, a list of layer names to which noise should be added,
a standard deviation sigma of the Gaussian noise, and an integer num_perturbations representing the number of times to apply 
noise to the specified layers.

For each perturbation, the function makes a deep copy of the original model and adds Gaussian noise to the specified layers
weights and biases using the add_noise function. The noisy weights and biases are then used to update the corresponding layer's 
weight and bias tensors.

After each perturbation, the modified model is added to a list, which is then returned once all perturbations have been applied.
"""

def add_gaussian_noise_to_model_brevitas(model, layer_names, sigma, num_perturbations, analog_noise_type):
    modified_models = []
    for _ in range(num_perturbations):
        modified_model = deepcopy(model)
        # add noise to the modified model
        for layer_name in layer_names:
            layer = getattr(modified_model, layer_name)
            with torch.no_grad():
                # Get the weight and bias tensors
                weight = layer.weight.cpu().detach().numpy()
                bias = layer.bias.cpu().detach().numpy() if layer.bias is not None else None
                # Add noise to the weight and bias tensors
                if (analog_noise_type):
                    noised_weight = add_gaussian_noise_independent(weight, sigma)
                    if bias is not None:
                        noised_bias = add_gaussian_noise_independent(bias, sigma)
                    # Update the layer's weight and bias tensors with the noised values
                    layer.weight = torch.nn.Parameter(
                        torch.tensor(noised_weight, dtype=torch.float))
                    if bias is not None:
                        layer.bias = torch.nn.Parameter(
                            torch.tensor(noised_bias, dtype=torch.float))
                else:
                    noised_weight = add_gaussian_noise_proportional(weight, sigma)
                    if bias is not None:
                        noised_bias = add_gaussian_noise_proportional(bias, sigma)
                    # Update the layer's weight and bias tensors with the noised values
                    layer.weight = torch.nn.Parameter(
                        torch.tensor(noised_weight, dtype=torch.float))
                    if bias is not None:
                        layer.bias = torch.nn.Parameter(
                            torch.tensor(noised_bias, dtype=torch.float))
                    

        modified_models.append(modified_model)

    return modified_models


"""
This function takes in a PyTorch model and modifies the weights and biases of specified layers by adding 
Gaussian noise with a given standard deviation. The function creates num_perturbations modified models, 
each with the same architecture as the original model but with different randomly perturbed weights and biases in the specified layers. 
The layers to be perturbed are specified by a list of layers, where each layer is identified by its index in the model's layers. 
The function returns a list of the modified models.
"""

def add_gaussian_noise_to_model_pytorch(model, layers, sigma, num_perturbations):
    
    # keep all modified models accuracy so we can get a correct average
    modified_models = []

    for _ in range(num_perturbations):    

        # create individual modified model
        modified_model = deepcopy(model)

        for i in range(len(layers)): 
            layer_name = f'layer{i + 1}'
            layer = getattr(modified_model, layer_name)
    
            # get weight and bias for layer
            weight_temp = layer[0].weight.data.cpu().detach().numpy()
            bias_temp = layer[0].bias.data.cpu().detach().numpy()
    
        
            # add noise to the weight and bias
            noise_w = add_noise(weight_temp, sigma)
            noise_b = add_noise(bias_temp, sigma)

            # place values back into the modified model for analysis
            layer[0].weight.data = torch.from_numpy(noise_w).float()
            layer[0].bias.data = torch.from_numpy(noise_b).float()

        # append this modified model to the list
        modified_models.append(modified_model)

        
    return modified_models


"""
This function gaussian_noise_plots_brevitas adds Gaussian noise to a given model for each layer and creates plots to visualize the effect of noise on test accuracy.

The function takes in the following parameters:

    num_perturbations: The number of times to perturb the model for each value of standard deviation.
    layer_names: A list of layer names to add noise to.
    sigma_vector: A list of standard deviation values to add noise with.
    model: The model to perturb.
    device: The device to use for computation.

The function first creates a folder to store the noise plots, if it does not already exist. Then, for each layer in layer_names, 
it loops over each value of standard deviation in sigma_vector and adds Gaussian noise to the model for the current layer only. 
It then tests the accuracy of each noisy model and appends the result to a list of test accuracies.

After testing for all standard deviation values, the function plots the test accuracy as a function of the standard deviation 
for the current layer and saves the plot to a file. It then computes the average test accuracy across all layers for each 
standard deviation value and plots the averaged test accuracies as a function of the standard deviation.

Finally, the function saves the average test accuracy plot to a file and displays it.
"""

def gaussian_noise_plots_brevitas(num_perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, analog_noise_type):
    
    if not os.path.exists("noise_plots_brevitas/gaussian_noise/"):
        os.makedirs("noise_plots_brevitas/gaussian_noise/")
    
    plt.style.use('default')
    
    
    # Create a list to store the test accuracies for all layers
    all_test_accs = []
    
    # Loop over each layer and plot the test accuracy as a function of the standard deviation for that layer
    for layer in layer_names:
        # Initialize a list to store the test accuracies for this layer
        test_accs = []
        # Iterate over the standard deviation values and add noise to the model for this layer only
        for sigma in sigma_vector:
            # Add noise to the model for the defined layer only
            noisy_models = add_gaussian_noise_to_model_brevitas(
                model, [layer], sigma, num_perturbations, analog_noise_type)
            accuracies = []
            # Test the accuracy of each noisy model and append the result to the accuracies list
            for noisy_model in noisy_models:
                # Move the model back to the target device
                noisy_model.to(device)
                accuracies.append(test(noisy_model, test_quantized_loader, device))
            # Calculate the average accuracy and print the result
            avg_accuracy = sum(accuracies) / len(accuracies)
            # Append the average accuracy to the test_accs list
            test_accs.append(avg_accuracy)
            
            if (analog_noise_type):
                print("Independent: Sigma Value: {}, Average Accuracy: {}%".format(
                    sigma, avg_accuracy))
            else:
                print("Proportional: Sigma Value: {}, Average Accuracy: {}%".format(
                    sigma, avg_accuracy))
                
        # Store the test accuracies for this layer in the all_test_accs list
        all_test_accs.append(test_accs)
        # Plot the test accuracies as a function of the standard deviation for this layer
        plt.plot(sigma_vector, test_accs,
                 label='{} Accuracy at Different Perturbation Levels'.format(layer))
        plt.xlabel('Standard Deviation')
        plt.ylabel('Test Accuracy')
        plt.title('Effect of Noise on Test Accuracy')
        plt.legend()
        
        if (analog_noise_type):
            plt.savefig("noise_plots_brevitas/gaussian_noise/{}_independent.png".format(layer))
        else:
            plt.savefig("noise_plots_brevitas/gaussian_noise/{}_proportional.png".format(layer))

        plt.clf()
        print('Done with Plot {}'.format(layer))
    # Compute the average test accuracy across all layers for each standard deviation value
    avg_test_accs = [sum(x) / len(x) for x in zip(*all_test_accs)]
    # Plot the averaged test accuracies as a function of the standard deviation
    plt.plot(sigma_vector, avg_test_accs, label='Average',
             linewidth=3, linestyle='--', color="black")
    plt.xlabel('Standard Deviation')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of Noise on Test Accuracy (Average)')
    plt.legend()
    plt.savefig("noise_plots_brevitas/gaussian_noise/average.png")
    plt.show()
    plt.clf()


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
