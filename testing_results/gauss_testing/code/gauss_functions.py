"""
Description: Each function is described individually and in full below.
Date: 24 August 2023

"""

## Imports
from datetime import datetime
import random
import os
import csv
import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt

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

#################################################
## Gaussian Noise Section
###############################################

"""
This function adds Gaussian noise to a given matrix. The sigma parameter controls
the standard deviation of the noise distribution. The function returns the original
matrix with added noise.
"""

def add_gaussian_noise_independent(matrix, sigma):

    random.seed(datetime.now().timestamp())

    noised_matrix = matrix + np.random.normal(0, scale=sigma)
    return noised_matrix

def add_gaussian_noise_proportional(matrix, sigma):

    random.seed(datetime.now().timestamp())

    noised_matrix = matrix * np.random.normal(1, scale=sigma)
    return noised_matrix


def return_noisy_matrix(independent, dim_matrix, sigma, device):

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

            random.seed(datetime.now().timestamp())

            layer = getattr(modified_model, layer_name)

            with torch.no_grad():

                if layer_name.lower().startswith("c"):
                    weight = layer.conv.weight.cpu().clone().detach().numpy()
                    bias = layer.conv.bias.cpu().detach().numpy() if layer.conv.bias is not None else None
                else:
                    weight = layer.weight.cpu().clone().detach().numpy()
                    bias = layer.bias.cpu().detach().numpy() if layer.bias is not None else None

                # Add noise to the weight and bias tensors
                if (analog_noise_type):
                    noised_weight = add_gaussian_noise_independent(weight, sigma)

                    if bias is not None:
                        noised_bias = add_gaussian_noise_independent(bias, sigma)
                else:
                    noised_weight = add_gaussian_noise_proportional(weight, sigma)

                    if bias is not None:
                        noised_bias = add_gaussian_noise_proportional(bias, sigma)


                # Update layer weights and bias tensors with noised values
                if layer_name.lower().startswith("c"):
                    layer.conv.weight = torch.nn.Parameter(torch.tensor(noised_weight, dtype = torch.float))

                    if bias is not None:
                        layer.conv.bias = torch.nn.Parameter(torch.tensor(noised_bias, dtype = torch.float))

                else:
                    layer.weight = torch.nn.Parameter(torch.tensor(noised_weight, dtype = torch.float))

                    if bias is not None:
                        layer.bias = torch.nn.Parameter(torch.tensor(noised_bias, dtype = torch.float))

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

def gaussian_noise_plots_brevitas(num_perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, analog_noise_type, model_name):

    if not os.path.exists(f"noise_plots_brevitas/gaussian_noise/{model_name}"):
        os.makedirs(f"noise_plots_brevitas/gaussian_noise/{model_name}")

    plt.style.use('default')

    # Create a list to store the test accuracies for all layers
    all_test_accs = []

    if (analog_noise_type):
        csv_file_path = os.path.join(f"noise_plots_brevitas/gaussian_noise/{model_name}", "raw_data_independent.csv")
    else:
        csv_file_path = os.path.join(f"noise_plots_brevitas/gaussian_noise/{model_name}", "raw_data_proportional.csv")



    # Create a CSV file to store the raw data
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Layer", "Sigma Value", "Average Accuracy"])

        # Loop over each layer and plot the test accuracy as a function of the standard deviation for that layer
        for layer in layer_names:

            ## Random seed
            random.seed(datetime.now().timestamp())

            print("Random Seed Generated")

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

                # Write the raw data to the CSV file
                writer.writerow([layer, sigma, avg_accuracy])

                if analog_noise_type:
                    print("Independent: Layer: {}, Sigma Value: {}, Average Accuracy: {}%".format(
                        layer, sigma, avg_accuracy))
                else:
                    print("Proportional: Layer: {}, Sigma Value: {}, Average Accuracy: {}%".format(
                        layer, sigma, avg_accuracy))

            # Store the test accuracies for this layer in the all_test_accs list
            all_test_accs.append(test_accs)
            # Plot the test accuracies as a function of the standard deviation for this layer
            plt.plot(sigma_vector, test_accs,
                     label='{} Accuracy'.format(layer))

        # Compute the average test accuracy across all layers for each standard deviation value
        avg_test_accs = [sum(x) / len(x) for x in zip(*all_test_accs)]

        # Plot the averaged test accuracies as a function of the standard deviation
        plt.plot(sigma_vector, avg_test_accs, label='Average',
                 linewidth=3, linestyle='--', color="black")

        plt.xlabel('Standard Deviation')
        plt.ylabel('Test Accuracy')
        plt.title('Effect of Noise on Test Accuracy (Individual Layers and Average)')
        plt.legend()

        if analog_noise_type:
            plt.savefig(f"noise_plots_brevitas/gaussian_noise/{model_name}/individual_and_average_independent.png")
        else:
            plt.savefig(f"noise_plots_brevitas/gaussian_noise/{model_name}/individual_and_average_proportional.png")

        plt.show()
        plt.clf()


def gaussian_noise_plots_brevitas_all(num_perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, analog_noise_type, model_name):

    if not os.path.exists(f"noise_plots_brevitas/gaussian_noise/{model_name}"):
        os.makedirs(f"noise_plots_brevitas/gaussian_noise/{model_name}")

    plt.style.use('default')

    # Create a CSV file to store the raw data
    if (analog_noise_type):
        csv_file_path = os.path.join(f"noise_plots_brevitas/gaussian_noise/{model_name}", "raw_data_independent_all_layers.csv")
    else:
        csv_file_path = os.path.join(f"noise_plots_brevitas/gaussian_noise/{model_name}", "raw_data_proportional_all_layers.csv")

    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Sigma Value", "Average Accuracy"])

        # Create a list to store the test accuracies for all layers
        all_test_accs = []

        random.seed(datetime.now().timestamp())

        print("Random Seed Generated")

        for sigma in sigma_vector:
            # Add noise to the model for all layers
            noisy_models = add_gaussian_noise_to_model_brevitas(
                model, layer_names, sigma, num_perturbations, analog_noise_type)
            accuracies = []

            for noisy_model in noisy_models:
                noisy_model.to(device)
                accuracies.append(test(noisy_model, test_quantized_loader, device))

            avg_accuracy = sum(accuracies) / len(accuracies)
            all_test_accs.append(avg_accuracy)

            # Write the raw data to the CSV file
            writer.writerow([sigma, avg_accuracy])

            if analog_noise_type:
                print("Independent: Sigma Value: {}, Average Accuracy: {}%".format(sigma, avg_accuracy))
            else:
                print("Proportional: Sigma Value: {}, Average Accuracy: {}%".format(sigma, avg_accuracy))

        # Plot the test accuracies as a function of the standard deviation for all layers
        plt.plot(sigma_vector, all_test_accs, label='Average Accuracy at Different Perturbation Levels')

        plt.xlabel('Standard Deviation')
        plt.ylabel('Test Accuracy')
        plt.title('Effect of Noise on Test Accuracy (All Layers)')
        plt.legend()

        if analog_noise_type:
            plt.savefig(f"noise_plots_brevitas/gaussian_noise/{model_name}/all_layers_independent.png")
        else:
            plt.savefig(f"noise_plots_brevitas/gaussian_noise/{model_name}/all_layers_proportional.png")

        plt.show()
        plt.clf()
