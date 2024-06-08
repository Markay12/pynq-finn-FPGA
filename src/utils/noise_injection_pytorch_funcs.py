"""
Noise Injection Utilities PyTorch
----------------------------------

Description:
    Utility functions for injecting noise into PyTorch models for testing their robustness.

Functions:
    add_digital_noise
    add_digital_noise_to_model_pytorch
    add_gaussian_noise_to_model_pytorch
    add_gaussian_noise_independent
    add_gaussian_noise_proportional
    ber_noise_plot_pytorch
    ber_noise_plot_multiple_layers_pytorch
    gaussian_noise_plot_pytorch
    gaussian_noise_plot_multiple_layers_pytorch

Usage:
    These utilities are intended to be imported and used in data processing or simulation scripts. Example of use:
    
        from noise_injection import function1_name

        modified_data = function1_name(data)

Notes:
    - These functions are optimized for performance and accuracy.
    
Author(s):
    Mark Ashinhust
    Christopher Bennett

Created on:
    05 December 2023

Last Modified:
    07 June 2024
"""

from copy import deepcopy
import csv
import datetime
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
import torch

# Adjust paths for importing necessary modules
current_path = Path(__file__).parent
parent_directory = current_path.parent
sys.path.append(str(parent_directory))

from test_noise import test
import setup

def add_digital_noise(matrix, ber):
    """
    Add digital noise to a PyTorch tensor.
    
    Args:
        matrix (torch.Tensor): The matrix to which noise will be added.
        ber (float): Bit Error Rate, determining the probability of each element in the matrix being affected by noise.

    Returns:
        torch.Tensor: The noisy matrix with values clipped to the range [-1, 1].
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("The input matrix must be a PyTorch tensor.")

    setup.gen_rand_seed()
    mask = torch.rand(matrix.size(), device=matrix.device) < ber
    noise = torch.rand(matrix.size(), device=matrix.device) * 2 - 1
    noisy_matrix = matrix + mask * noise
    return torch.clamp(noisy_matrix, -1, 1)

def add_digital_noise_to_model_pytorch(model, layer_names, ber, num_perturbations):
    """
    Add digital noise to specified layers of a PyTorch model.

    Args:
        model (torch.nn.Module): The neural network model.
        layer_names (list of str): Names of the layers to receive digital noise.
        ber (float): Bit Error Rate.
        num_perturbations (int): Number of perturbed models to generate.

    Returns:
        list of torch.nn.Module: List of models with noise added.
    """
    modified_models = []

    for _ in range(num_perturbations):
        modified_model = deepcopy(model)
        for layer_name in layer_names:
            layer = getattr(modified_model, layer_name)
            if layer_name.lower().startswith("c"):
                weight_tensor = layer.conv.weight.detach().clone()
            else:
                weight_tensor = layer.weight.detach().clone()
            noisy_weight = add_digital_noise(weight_tensor, ber)
            with torch.no_grad():
                if layer_name.lower().startswith("c"):
                    layer.conv.weight = torch.nn.Parameter(noisy_weight)
                else:
                    layer.weight = torch.nn.Parameter(noisy_weight)
        modified_models.append(modified_model)

    return modified_models

def add_gaussian_noise_independent(matrix, sigma):
    """
    Add independent Gaussian noise to a matrix.

    Args:
        matrix (np.ndarray): The matrix to which noise will be added.
        sigma (float): Standard deviation of the Gaussian noise distribution.

    Returns:
        np.ndarray: The noisy matrix.
    """
    setup.gen_rand_seed()
    noise = np.random.normal(0, scale=sigma, size=matrix.shape)
    return matrix + noise

def add_gaussian_noise_proportional(matrix, sigma):
    """
    Add proportional Gaussian noise to a matrix.

    Args:
        matrix (np.ndarray): The matrix to which noise will be added.
        sigma (float): Standard deviation of the Gaussian noise distribution.

    Returns:
        np.ndarray: The noisy matrix.
    """
    setup.gen_rand_seed()
    noise = np.random.normal(1, scale=sigma, size=matrix.shape)
    return matrix * noise

def add_gaussian_noise_to_model_pytorch(model, layer_names, sigma, num_perturbations, analog_noise_type):
    """
    Add Gaussian noise to specified layers of a PyTorch model.

    Args:
        model (torch.nn.Module): The neural network model.
        layer_names (list of str): Names of the layers to receive Gaussian noise.
        sigma (float): Standard deviation of the Gaussian noise distribution.
        num_perturbations (int): Number of perturbed models to generate.
        analog_noise_type (bool): True for independent Gaussian noise, False for proportional Gaussian noise.

    Returns:
        list of torch.nn.Module: List of models with noise added.
    """
    modified_models = []

    for _ in range(num_perturbations):
        modified_model = deepcopy(model)
        for layer_name in layer_names:
            layer = getattr(modified_model, layer_name)
            with torch.no_grad():
                if layer_name.lower().startswith("c"):
                    weight = layer.conv.weight.cpu().clone().detach().numpy()
                    bias = layer.conv.bias.cpu().detach().numpy() if layer.conv.bias is not None else None
                else:
                    weight = layer.weight.cpu().clone().detach().numpy()
                    bias = layer.bias.cpu().detach().numpy() if layer.bias is not None else None

                if analog_noise_type:
                    noised_weight = add_gaussian_noise_independent(weight, sigma)
                    if bias is not None:
                        noised_bias = add_gaussian_noise_independent(bias, sigma)
                else:
                    noised_weight = add_gaussian_noise_proportional(weight, sigma)
                    if bias is not None:
                        noised_bias = add_gaussian_noise_proportional(bias, sigma)

                if layer_name.lower().startswith("c"):
                    layer.conv.weight = torch.nn.Parameter(torch.tensor(noised_weight, dtype=torch.float))
                    if bias is not None:
                        layer.conv.bias = torch.nn.Parameter(torch.tensor(noised_bias, dtype=torch.float))
                else:
                    layer.weight = torch.nn.Parameter(torch.tensor(noised_weight, dtype=torch.float))
                    if bias is not None:
                        layer.bias = torch.nn.Parameter(torch.tensor(noised_bias, dtype=torch.float))

        modified_models.append(modified_model)

    return modified_models

def ber_noise_plot_pytorch(num_perturbations, layer_names, ber_vector, model, device, test_quantized_loader, model_name):
    """
    Plot the effect of BER noise on the test accuracy of a model.

    Args:
        num_perturbations (int): Number of perturbed models to generate for each BER value.
        layer_names (list of str): Names of the layers to receive digital noise.
        ber_vector (list of float): List of BER values to apply as noise.
        model (torch.nn.Module): The neural network model.
        device (torch.device): The device to run the operations on.
        test_quantized_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        model_name (str): Name of the model for organizing output files and plots.
    """
    output_dir = f"noise_plots_pytorch/ber_noise_pytorch/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('default')
    all_test_accs = []
    csv_file_path = os.path.join(output_dir, "raw_data.csv")

    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Layer", "BER Value", "Average Accuracy"])

        for layer in layer_names:
            test_accs = []
            for ber in ber_vector:
                noisy_models = add_digital_noise_to_model_pytorch(model, [layer], ber, num_perturbations)
                accuracies = [test(noisy_model.to(device), test_quantized_loader, device) for noisy_model in noisy_models]
                avg_accuracy = sum(accuracies) / len(accuracies)
                test_accs.append(avg_accuracy)
                writer.writerow([layer, ber, avg_accuracy])
                print(f"Layer: {layer}\tBER Value: {ber}\tAverage Accuracy: {avg_accuracy}")

            all_test_accs.append(test_accs)
            plt.plot(ber_vector, test_accs, label=f'{layer} Accuracy at Different Perturbation Levels')

    avg_test_accs = [sum(x) / len(x) for x in zip(*all_test_accs)]
    plt.plot(ber_vector, avg_test_accs, label='Average', linewidth=3, linestyle='--', color="black")
    plt.xlabel('BER Value')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of BER Noise on Test Accuracy (Individual Layers and Average)')
    plt.legend(title='Layers')
    plt.savefig(os.path.join(output_dir, "individual_and_average.png"))
    plt.show()
    plt.clf()

def ber_noise_plot_multiple_layers_pytorch(num_perturbations, layer_names, ber_vector, model, device, test_quantized_loader, model_name):
    """
    Plot the effect of BER noise on the test accuracy of a model across multiple layers.

    Args:
        num_perturbations (int): Number of perturbed models to generate for each BER value.
        layer_names (list of str): Names of the layers to receive digital noise.
        ber_vector (list of float): List of BER values to apply as noise.
        model (torch.nn.Module): The neural network model.
        device (torch.device): The device to run the operations on.
        test_quantized_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        model_name (str): Name of the model for organizing output files and plots.
    """
    output_dir = f"noise_plots_pytorch/ber_noise_pytorch/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('default')
    test_accs = []
    csv_file_path = os.path.join(output_dir, "raw_data_all_layers.csv")

    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["BER Value", "Average Accuracy"])

        for ber in ber_vector:
            noisy_models = add_digital_noise_to_model_pytorch(model, layer_names, ber, num_perturbations)
            accuracies = [test(noisy_model.to(device), test_quantized_loader, device) for noisy_model in noisy_models]
            avg_accuracy = sum(accuracies) / len(accuracies)
            test_accs.append(avg_accuracy)
            writer.writerow([ber, avg_accuracy])
            print(f"BER Value: {ber}\tAverage Accuracy: {avg_accuracy}")

    plt.plot(ber_vector, test_accs, label='Accuracy at Different Perturbation Levels for all Layers')
    plt.xlabel('BER Value')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of BER Noise on Test Accuracy (All Layers)')
    plt.legend(title='Layers')
    plt.savefig(os.path.join(output_dir, "all_layers.png"))
    plt.show()
    plt.clf()

def gaussian_noise_plot_pytorch(num_perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, analog_noise_type, model_name):
    """
    Plot the effect of Gaussian noise on the test accuracy of a model for individual layers.

    Args:
        num_perturbations (int): Number of modified models to generate for each value of standard deviation.
        layer_names (list of str): Names of the layers to receive Gaussian noise.
        sigma_vector (list of float): List of standard deviation values for the Gaussian noise.
        model (torch.nn.Module): The neural network model.
        device (torch.device): The device to run the operations on.
        test_quantized_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        analog_noise_type (bool): True for independent Gaussian noise, False for proportional Gaussian noise.
        model_name (str): Name of the model for organizing output files and plots.
    """
    output_dir = f"noise_plots_pytorch/gaussian_noise_pytorch/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('default')
    all_test_accs = []
    csv_file_path = os.path.join(output_dir, f"raw_data_{'independent' if analog_noise_type else 'proportional'}.csv")

    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Layer", "Sigma Value", "Average Accuracy"])

        for layer in layer_names:
            test_accs = []

            for sigma in sigma_vector:
                noisy_models = add_gaussian_noise_to_model_pytorch(model, [layer], sigma, num_perturbations, analog_noise_type)
                accuracies = [test(noisy_model.to(device), test_quantized_loader, device) for noisy_model in noisy_models]
                avg_accuracy = sum(accuracies) / len(accuracies)
                test_accs.append(avg_accuracy)
                writer.writerow([layer, sigma, avg_accuracy])
                noise_type_str = "Independent" if analog_noise_type else "Proportional"
                print(f"{noise_type_str}: Layer: {layer}, Sigma Value: {sigma}, Average Accuracy: {avg_accuracy}")

            all_test_accs.append(test_accs)
            plt.plot(sigma_vector, test_accs, label=f'{layer} Accuracy at Different Perturbation Levels')

        avg_test_accs = [sum(x) / len(x) for x in zip(*all_test_accs)]
        plt.plot(sigma_vector, avg_test_accs, label='Average', linewidth=3, linestyle='--', color="black")
        plt.xlabel('Standard Deviation')
        plt.ylabel('Test Accuracy')
        plt.title('Effect of Noise on Test Accuracy (Individual Layers and Average)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"individual_and_average_{'independent' if analog_noise_type else 'proportional'}.png"))
        plt.show()
        plt.clf()

def gaussian_noise_plot_multiple_layers_pytorch(num_perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, analog_noise_type, model_name):
    """
    Plot the effect of Gaussian noise on the test accuracy of a model across multiple layers.

    Args:
        num_perturbations (int): Number of modified models to generate for each value of standard deviation.
        layer_names (list of str): Names of the layers to receive Gaussian noise.
        sigma_vector (list of float): List of standard deviation values for the Gaussian noise.
        model (torch.nn.Module): The neural network model.
        device (torch.device): The device to run the operations on.
        test_quantized_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        analog_noise_type (bool): True for independent Gaussian noise, False for proportional Gaussian noise.
        model_name (str): Name of the model for organizing output files and plots.
    """
    output_dir = f"noise_plots_pytorch/gaussian_noise_pytorch/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('default')
    csv_file_path = os.path.join(output_dir, f"raw_data_{'independent_all_layers' if analog_noise_type else 'proportional_all_layers'}.csv")

    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Sigma Value", "Average Accuracy"])
        all_test_accs = []

        for sigma in sigma_vector:
            noisy_models = add_gaussian_noise_to_model_pytorch(model, layer_names, sigma, num_perturbations, analog_noise_type)
            accuracies = [test(noisy_model.to(device), test_quantized_loader, device) for noisy_model in noisy_models]
            avg_accuracy = sum(accuracies) / len(accuracies)
            all_test_accs.append(avg_accuracy)
            writer.writerow([sigma, avg_accuracy])
            noise_type_str = "Independent" if analog_noise_type else "Proportional"
            print(f"{noise_type_str}: Sigma Value: {sigma}, Average Accuracy: {avg_accuracy}")

        plt.plot(sigma_vector, all_test_accs, label='Average Accuracy at Different Perturbation Levels')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Test Accuracy')
        plt.title('Effect of Noise on Test Accuracy (All Layers)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"all_layers_{'independent' if analog_noise_type else 'proportional'}.png"))
        plt.show()
        plt.clf()

def select_test(perturbations, layer_names, ber_vals, model, device, val_quantized_loader, model_name, sigma_vector_prop, sigma_vector_ind):
    """
    Prompt the user to select tests and run the selected tests.

    Args:
        perturbations (int): Number of perturbations.
        layer_names (list of str): List of layer names.
        ber_vals (list of float): List of BER values.
        model (torch.nn.Module): The neural network model.
        device (torch.device): The device to run the operations on.
        val_quantized_loader (torch.utils.data.DataLoader): DataLoader containing the validation dataset.
        model_name (str): Name of the model for organizing output files and plots.
        sigma_vector_prop (list of float): List of proportional sigma values for Gaussian noise.
        sigma_vector_ind (list of float): List of independent sigma values for Gaussian noise.
    """
    print("Please select the tests you would like to run:")
    print("1. PyTorch BER Noise Injection (Individual Layers)")
    print("2. PyTorch BER Noise Injection (Multiple Layers)")
    print("3. PyTorch Gaussian Noise Injection (Individual Layers, Proportional)")
    print("4. PyTorch Gaussian Noise Injection (Individual Layers, Independent)")
    print("5. PyTorch Gaussian Noise Injection (Multiple Layers, Proportional)")
    print("6. PyTorch Gaussian Noise Injection (Multiple Layers, Independent)")

    selected_tests = input("Enter the numbers of the tests to run, separated by commas (e.g., 1,3,5): ")
    selected_tests = [int(test.strip()) for test in selected_tests.split(',')]

    for test_number in selected_tests:
        if test_number == 1:
            setup.print_header("Beginning PyTorch BER Noise Injection", subtitle="Individual Layers")
            ber_noise_plot_pytorch(perturbations, layer_names, ber_vals, model, device, val_quantized_loader, model_name)
        elif test_number == 2:
            setup.print_header("Beginning PyTorch BER Noise Injection", None, multiple_layers=True)
            ber_noise_plot_multiple_layers_pytorch(perturbations, layer_names, ber_vals, model, device, val_quantized_loader, model_name)
        elif test_number == 3:
            setup.print_header("Beginning PyTorch Gaussian Noise Injection", subtitle="Individual Layers, Proportional")
            gaussian_noise_plot_pytorch(perturbations, layer_names, sigma_vector_prop, model, device, val_quantized_loader, False, model_name)
        elif test_number == 4:
            setup.print_header("Beginning PyTorch Gaussian Noise Injection", subtitle="Individual Layers, Independent")
            gaussian_noise_plot_pytorch(perturbations, layer_names, sigma_vector_ind, model, device, val_quantized_loader, True, model_name)
        elif test_number == 5:
            setup.print_header("Beginning PyTorch Gaussian Noise Injection", subtitle="Proportional", multiple_layers=True)
            gaussian_noise_plot_multiple_layers_pytorch(perturbations, layer_names, sigma_vector_prop, model, device, val_quantized_loader, False, model_name)
        elif test_number == 6:
            setup.print_header("Beginning PyTorch Gaussian Noise Injection", subtitle="Independent", multiple_layers=True)
            gaussian_noise_plot_multiple_layers_pytorch(perturbations, layer_names, sigma_vector_ind, model, device, val_quantized_loader, True, model_name)
        else:
            print(f"Invalid selection: {test_number}")
