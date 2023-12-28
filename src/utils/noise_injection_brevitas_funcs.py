"""
Noise Injection Utilities
-------------------------

Description:
    This module contains functions for injecting noise into data. These functions
    are designed to simulate various types of noise conditions to test the robustness
    and performance of data processing algorithms.

Functions:
    add_digital_noise()
    add_digital_noise_to_model_brevitas()
    add_gaussian_noise_independent()
    add_gaussian_noise_proportional()
    add_gaussian_noise_to_model_brevitas()
    add_gaussian_noise_to_model_pytorch()
    ber_noise_plot_brevitas()
    ber_noise_plot_multiple_layers_brevitas()
    gaussian_noise_plots_brevitas()
    gaussian_noise_plots_brevitas_all()
    return_noisy_matrix()

Usage:
    These utilities are intended to be imported and used in data processing or
    simulation scripts. Example of use:
    
        from noise_injection import function1_name

        modified_data = function1_name(data)

Notes:
    - These functions are optimized for performance and accuracy.
    
Author(s):
    Mark Ashinhust
    Christopher Bennett

Created on:
    05 December 2024

Last Modified:
    08 December 2024

"""

## IMPORT STATEMENTS
from copy import deepcopy
import datetime
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import sys
import torch

# Current script path
current_path = Path(__file__).parent

# Get parent directory
parent_directory = current_path

print(parent_directory)

# Add parent to the Python path
sys.path.append( str( parent_directory ) )

from test_noise import test

"""
Function Name: add_digital_noise()

Parameters:
    1. matrix
        Description:    The matrix to which noise will be added.
        Type:           numpy.ndarray.
        Details:        This matrix represents the data (such as weights of a neural network layer) 
                        to which digital noise is to be applied.

    2. ber
        Description:    Bit Error Rate, determining the probability of each element in the 
                        matrix being affected by noise.
        Type:           Float.
        Details:        BER controls the amount of noise that will be added to the matrix. 
                        It represents the likelihood of any given element in the matrix to be 
                        modified by noise.

Function Process:
    - A new random seed is generated for each invocation of the function, based on the 
      current datetime. This seed ensures that the noise pattern is different each time 
      the function is called.
    - A boolean mask is created, the same shape as the input matrix, where each element is 
      'True' with a probability defined by the BER. This mask determines which elements of 
      the matrix will be affected by noise.
    - A noise matrix, with random values between -1 and 1, is generated and applied to the 
      original matrix wherever the mask is 'True'.
    - The resulting matrix, now with added noise, is then clipped to ensure all values 
      remain within the range of [-1, 1].

Return:
    - numpy.ndarray: The input matrix after adding digital noise and clipping its values 
                     to the range of [-1, 1].
 
"""

def add_digital_noise( matrix, ber ):

    print( "Adding Digital Noise - START" )

    # Generate a new seed each time to add noise to the model.
    # This creates more accurate randomess and noise generation.
    current_time = datetime.datetime.now()
    seed = int( current_time.timestamp() )

    # Set the seed 
    random.seed( seed );
    print( "New seed has been generated." );

    # Generate a boolean mask with the same shape as the matrix where each element is True with probability `ber`
    mask = np.random.rand( *matrix.shape ) < ber
    
    # Generate a matrix of random values between -1 and 1 with the same shape as the input matrix
    noise = np.random.uniform( -1, 1, size=matrix.shape )
    
    # Add the noise to the input matrix only where the mask is True
    noisy_matrix = matrix + mask * noise

    print( "Digital Noise Added - SUCCESSFUL - END" )
    
    # Clip the noisy matrix to the range [-1, 1] to ensure that all elements are within this range
    return np.clip( noisy_matrix, -1, 1 )

"""
Function Name: add_digital_noise_to_model_brevitas()

Paramters:
    1. model
        Description:        The neural network model (or specific layers of the model) to which digital noise will be added.
        Type:               Brevitas layer or similar PyTorch model object.
        Details:            This is the primary object of the function, where digital noise will be applied to 
                            specified layers.

    2. layer_names
        Description:        Names of the layers within the model that will receive the digital noise.
        Type:               List of strings.
        Details:            Each string in this list should correspond to the name of a layer in the model. 
                            The function applies digital noise to the weights of these layers.

    3. ber
        Description:        Bit Error Rate, a parameter defining the rate at which digital noise affects the layer's weights.
        Type:               Float.
        Details:            This parameter controls the intensity or probability of digital noise application. 
                            It's a key factor in determining how the weights of the model are perturbed.

    4. num_perturbations
        Description:        The number of perturbed models to generate.
        Type:               Integer.
        Details:            Specifies how many times the function will create a modified version of the model with 
                            digital noise applied. Each iteration produces a new model with independently applied noise.

The function add_digital_noise_to_brevitas_layer adds digital noise to the weight tensor of a brevitas layer. 
It first gets the weight tensor of the layer and converts it to a numpy array. Then, it applies digital noise 
to the weight tensor using the add_digital_noise function. The noisy weight tensor is converted back to a 
PyTorch tensor and is used to update the layer's weight tensor.
"""

def add_digital_noise_to_model_brevitas( model, layer_names, ber, num_perturbations ):

    print( "Adding Digital Noise to Model - BREVITAS - START" )
    
    modified_models = []
    
    for _ in range(num_perturbations):
        
        modified_model = deepcopy(model)
        
        for layer_name in layer_names:
            
            if layer_name.lower().startswith("c"):
                layer = getattr(modified_model, layer_name)
                weight_tensor = layer.conv.weight.cpu().clone().detach()
            else:
                layer = getattr(modified_model, layer_name)
                weight_tensor = layer.weight.cpu().clone().detach()

            #print(f"Original weights for {layer_name}: {weight_tensor[0,0,0,:]}")

            with torch.no_grad():
                
                noisy_weight = add_digital_noise(weight_tensor.numpy(), ber)
                
                #print(f"Noisy weights for {layer_name}: {noisy_weight[0,0,0,:]}")

                if layer_name.lower().startswith("c"):
                    layer.conv.weight = torch.nn.Parameter(torch.tensor(noisy_weight, dtype=torch.float))
                else:
                    layer.weight = torch.nn.Parameter(torch.tensor(noisy_weight, dtype=torch.float))
                
        modified_models.append(modified_model)

    print( "Digital Noise Added to Model - SUCCESSFUL - END" )
        
    return modified_models

"""
Function Name: add_gaussian_noise_independent()

Parameters: 

    1. matrix:
        Description:    The matrix to which Gaussian noise will be added.
        Type:           NumPy array or similar matrix.
        Details:        This is the primary data to which noise will be applied.

    2. sigma:
        Description:    Standard deviation of the Gaussian noise distribution.
        Type:           Float.
        Details:        This parameter controls the intensity of the noise. A higher sigma value 
                        results in more variation in the noise added to matrix.

This function adds Gaussian noise to a given matrix. The sigma parameter controls
the standard deviation of the noise distribution. The function returns the original
matrix with added noise.
"""

def add_gaussian_noise_independent( matrix, sigma ):

    print( "Generating Independent Gaussian Noise - START" )

    # Generate a new seed each time to add noise to the model.
    # This creates more accurate randomess and noise generation.
    current_time = datetime.datetime.now()
    seed = int( current_time.timestamp() )

    # Set the seed 
    random.seed( seed );
    print( "New seed has been generated." );

    noised_matrix = matrix + np.random.normal( 0, scale=sigma )

    print( "Independent Gaussian Noise Matrix Generated - SUCCESSFUL - END" )

    return noised_matrix

"""
Function Name: add_gaussian_noise_proportional()

Parameters: 

    1.  matrix
        Description:    The matrix to be modified with proportional Gaussian noise.
        Type:           NumPy array or similar matrix.
        Details:        This matrix will be element-wise multiplied by a Gaussian noise factor.

    2. sigma
        Description:    Standard deviation of the Gaussian noise distribution.
        Type:           Float.
        Details:        Determines the variation in the proportional noise. 
                        Unlike the independent noise addition, this affects the matrix elements in a multiplicative manner.
"""

def add_gaussian_noise_proportional( matrix, sigma ):

    print( "Generating Proportional Gaussian Noise - START" )

    # Generate a new seed each time to add noise to the model.
    # This creates more accurate randomess and noise generation.
    current_time = datetime.datetime.now()
    seed = int( current_time.timestamp() )

    # Set the seed 
    random.seed( seed );
    print( "New seed has been generated." );

    noised_matrix = matrix * np.random.normal( 1, scale=sigma )

    print( "Proportional Gaussian Noise Matrix Generated - SUCCESSFUL - END" )

    return noised_matrix

"""
Function Name: add_gaussian_noise_to_model_brevitas()

Paramters:

    1.  model:
            Description:        The neural network model to which Gaussian noise will be added.
            Type:               Brevitas layer or similar PyTorch model object.
            Details:            This is the primary object of the function, where Gaussian noise will be applied to 
                                specified layers.

    2.  layer_names:
            Description:        Names of the layers within the model that will receive the Gaussian noise.
            Type:               List of strings.
            Details:            Each string in this list should correspond to the name of a layer in the model. 
                                The function applies Gaussian noise to the weights (and biases, if applicable) of these layers.

    3.  sigma:
            Description:        Standard deviation of the Gaussian noise distribution.
            Type:               Float.
            Details:            This parameter controls the intensity of the noise. A higher sigma value results in more variation in the noise added to the weights and biases of the layers.

    4.  num_perturbations:
            Description:        The number of perturbed models to generate.
            Type:               Integer.
            Details:            Specifies how many times the function will create a modified version of the model with 
                                Gaussian noise applied.

    5.  analog_noise_type:
            Description:        Flag to determine the type of Gaussian noise addition.
            Type:               Boolean.
            Details:            If True, independent Gaussian noise is added; if False, proportional Gaussian noise is added. 
                                This flag decides whether the noise is added independently to each element of the weight and 
                                bias tensors (using add_gaussian_noise_independent) or in a proportional manner 
                                (using add_gaussian_noise_proportional).

The add_gaussian_noise_to_model_brevitas function takes a Brevitas model, a list of layer names to which noise should be added,
a standard deviation sigma of the Gaussian noise, and an integer num_perturbations representing the number of times to apply 
noise to the specified layers.

For each perturbation, the function makes a deep copy of the original model and adds Gaussian noise to the specified layers
weights and biases using the add_noise function. The noisy weights and biases are then used to update the corresponding layer's 
weight and bias tensors.

After each perturbation, the modified model is added to a list, which is then returned once all perturbations have been applied.
"""

def add_gaussian_noise_to_model_brevitas( model, layer_names, sigma, num_perturbations, analog_noise_type ):

    print( "Adding Gaussian Noise to Model - BREVITAS - START" )

    modified_models = []

    for _ in range( num_perturbations ):

        modified_model = deepcopy( model )

        # add noise to the modified model
        for layer_name in layer_names:

            layer = getattr( modified_model, layer_name )

            with torch.no_grad():

                # Get the weight and bias tensors
                weight = layer.weight.cpu().detach().numpy()
                bias = layer.bias.cpu().detach().numpy() if layer.bias is not None else None

                print( "Adding noise for the model at Layer: ", layer_name, " and Perturbation: ", _ );

                # Add noise to the weight and bias tensors
                if ( analog_noise_type ):
                    noised_weight = add_gaussian_noise_independent( weight, sigma )

                    if bias is not None:
                        noised_bias = add_gaussian_noise_independent( bias, sigma )

                    # Update the layer's weight and bias tensors with the noised values
                    layer.weight = torch.nn.Parameter(
                        torch.tensor( noised_weight, dtype=torch.float ) )
                    
                    if bias is not None:
                        layer.bias = torch.nn.Parameter(
                            torch.tensor( noised_bias, dtype=torch.float ) )
                else:
                    noised_weight = add_gaussian_noise_proportional( weight, sigma )

                    if bias is not None:
                        noised_bias = add_gaussian_noise_proportional( bias, sigma )

                    # Update the layer's weight and bias tensors with the noised values
                    layer.weight = torch.nn.Parameter(
                        torch.tensor( noised_weight, dtype=torch.float ) )
                    
                    if bias is not None:
                        layer.bias = torch.nn.Parameter(
                            torch.tensor( noised_bias, dtype=torch.float ) )
                    
        modified_models.append( modified_model )

    print( "Gaussian Noise Added to Model - BREVITAS - END" )

    return modified_models

"""
Function Name: add_gaussian_noise_to_model_pytorch()

Parameters:

    1.  model:
            Description:    The PyTorch model whose weights and biases are to be modified.
            Type:           PyTorch model object.
            Details:        This is the original model that will serve as a template for creating perturbed copies. 
                            The function does not modify this model directly but creates deep copies of it.

    2.  layers:
            Description:    A list specifying which layers of the model are to be modified.
            Type:           List of integers.
            Details:        Each element in this list is an index that corresponds to a layer in the model. These layers are the targets for noise addition. The function assumes a naming convention like layer1, layer2, etc., in the model.

    3.  sigma:
            Description:    Standard deviation of the Gaussian noise to be added.
            Type:           Float.
            Details:        Determines the intensity of the Gaussian noise. This value controls how much the weights and biases are perturbed from their original values.

    4.  num_perturbations:
            Description:    The number of modified models to generate.
            Type:           Integer.
            Details:        Indicates how many copies of the original model, each with differently perturbed weights and biases, are to be created.

This function takes in a PyTorch model and modifies the weights and biases of specified layers by adding 
Gaussian noise with a given standard deviation. The function creates num_perturbations modified models, 
each with the same architecture as the original model but with different randomly perturbed weights and biases in the specified layers. 
The layers to be perturbed are specified by a list of layers, where each layer is identified by its index in the model's layers. 
The function returns a list of the modified models.
"""

def add_gaussian_noise_to_model_pytorch( model, layers, sigma, num_perturbations ):

    print( "Adding Gaussian Noise to Model - PyTorch - START" )
    
    # keep all modified models accuracy so we can get a correct average
    modified_models = []

    for _ in range( num_perturbations ):    

        # create individual modified model
        modified_model = deepcopy( model )

        for i in range( len( layers ) ): 
            layer_name = f'layer{i + 1}'
            layer = getattr( modified_model, layer_name )
    
            # get weight and bias for layer
            weight_temp = layer[0].weight.data.cpu().detach().numpy()
            bias_temp = layer[0].bias.data.cpu().detach().numpy()
    
            # add noise to the weight and bias
            # need to work on this to add independent vs. proportional noise
            noise_w = ( weight_temp, sigma )
            noise_b = ( bias_temp, sigma )

            # place values back into the modified model for analysis
            layer[0].weight.data = torch.from_numpy(noise_w).float()
            layer[0].bias.data = torch.from_numpy(noise_b).float()

        # append this modified model to the list
        modified_models.append(modified_model)

    print( "Gaussian Noise Added to Model - PyTorch - SUCCESSFUL - END" )

    return modified_models

"""
Function Name: ber_noise_plot_brevitas()

Paramters:

    1.  num_perturbations
        Description:        The number of perturbed models to generate for each BER value.
        Type:               Integer.
        Details:            This parameter dictates how many times the function will create a modified version 
                            of the model with digital noise applied for each BER value.

    2.  layer_names
        Description:        Names of the layers within the model that will be subjected to digital noise.
        Type:               List of strings.
        Details:            Each string in this list should correspond to the name of a layer in the model. 
                            The function applies digital noise to these specified layers.

    3.  ber_vector
        Description:        A list of Bit Error Rate (BER) values to apply as noise.
        Type:               List of floats.
        Details:            Each BER value in this vector represents a different level of noise to be applied to the model's 
                            layers. The function iterates over these values to evaluate the effect of different noise 
                            intensities.

    4. model
        Description:        The neural network model to be evaluated under different noise conditions.
        Type:               Brevitas layer or similar PyTorch model object.
        Details:            This is the base model on which the digital noise will be applied for testing its performance.

    5.  device
        Description:        The computational device (e.g., CPU, GPU) where the operation is to be performed.
        Type:               PyTorch device object.
        Details:            Ensures that the model and all operations are performed on the specified device.

    6.  test_quantized_loader
        Description:        DataLoader containing the test dataset for evaluating the model.
        Type:               PyTorch DataLoader object.
        Details:            This DataLoader is used to test the model's performance (accuracy) under various noise conditions.

    7.  model_name:
        Description:        Name of the model, used for organizing output files and plots.
        Type:               String.
        Details:            This name is used to create directories and name files where the results (plots and data) 
                            will be saved. It helps in identifying and segregating the results for different models or 
                            experimental setups.

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

def ber_noise_plot_brevitas(num_perturbations, layer_names, ber_vector, model, device, test_quantized_loader, model_name):

    print( "Generating Bit Error Rate Noise Plots - START" )

    if not os.path.exists(f"noise_plots_brevitas/ber_noise/{model_name}"):
        os.makedirs(f"noise_plots_brevitas/ber_noise/{model_name}")
    
    plt.style.use('default')
    
    all_test_accs = []

    # Create a CSV file to store the raw data
    csv_file_path = os.path.join( f"noise_plots_brevitas/ber_noise/{model_name}", "raw_data.csv" )

    with open( csv_file_path, mode='w', newline='' ) as csv_file:

        writer = csv.writer( csv_file )
        writer.writerow( [ "Layer", "BER Value", "Average Accuracy" ] )

        for layer in layer_names:
            test_accs = []
            
            for ber in ber_vector:
                noisy_models = add_digital_noise_to_model_brevitas( model, [layer], ber, num_perturbations )
                
                accuracies = []
                
                for noisy_model in noisy_models:
                    noisy_model.to(device)
                    accuracies.append(test(noisy_model, test_quantized_loader, device))
                
                avg_accuracy = sum(accuracies) / len(accuracies)
                test_accs.append(avg_accuracy)

                # Write the raw data to the CSV file
                writer.writerow([layer, ber, avg_accuracy])

                print("Layer: {}\tBER Value: {}\tAverage Accuracy: {}".format(layer, ber, avg_accuracy))
            
            all_test_accs.append(test_accs)
            
            plt.plot(ber_vector, test_accs, label='{} Accuracy at Different Perturbation Levels'.format(layer))

    avg_test_accs = [sum(x) / len(x) for x in zip(*all_test_accs)]

    print( "Noise Plot Generated - SAVING - END" )

    plt.plot(ber_vector, avg_test_accs, label='Average', linewidth=3, linestyle='--', color="black")

    plt.xlabel('BER Value')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of BER Noise on Test Accuracy (Individual Layers and Average)')
    plt.legend(title='Layers')
    plt.savefig(f"noise_plots_brevitas/ber_noise/{model_name}/individual_and_average.png")
    plt.show()
    plt.clf()

    print( "Plot Generated and Saved - END" )

"""
Function Name: ber_noise_plot_multiple_layers_brevitas()

Parameters:

    1.  num_perturbations
            Description:        The number of perturbed models to generate for each BER value.
            Type:               Integer.
            Details:            This parameter dictates how many times the function will create a modified version 
                                of the model with digital noise applied for each BER value.

    2.  layer_names
            Description:        Names of the layers within the model that will be subjected to digital noise.
            Type:               List of strings.
            Details:            Each string in this list should correspond to the name of a layer in the model. The function applies digital noise to these specified layers.

    3.  ber_vector:
            Description:        A list of Bit Error Rate (BER) values to apply as noise.
            Type:               List of floats.
            Details:            Each BER value in this vector represents a different level of noise to be applied to the 
                                model's layers. The function iterates over these values to evaluate the effect of different 
                                noise intensities.

    4.  model
            Description:        The neural network model to be evaluated under different noise conditions.
            Type:               Brevitas layer or similar PyTorch model object.
            Details:            This is the base model on which the digital noise will be applied for testing its performance.

    5.  device
            Description:        The computational device (e.g., CPU, GPU) where the operation is to be performed.
            Type:               PyTorch device object.
            Details:            Ensures that the model and all operations are performed on the specified device.

    6.  test_quantized_loader
            Description:        DataLoader containing the test dataset for evaluating the model.
            Type:               PyTorch DataLoader object.
            Details:            This DataLoader is used to test the model's performance (accuracy) under various noise 
                                conditions.

    7.  model_name:
            Description:        Name of the model, used for organizing output files and plots.
            Type:               String.
            Details:            This name is used to create directories and name files where the results (plots and data) 
                                will be saved. It helps in identifying and segregating the results for different models or 
                                experimental setups.
"""

def ber_noise_plot_multiple_layers_brevitas(num_perturbations, layer_names, ber_vector, model, device, test_quantized_loader, model_name):

    print( "Generating Bit Error Rate Noise Plot for Multiple Layers - BREVITAS - START" )

    if not os.path.exists(f"noise_plots_brevitas/ber_noise/{model_name}"):
        os.makedirs(f"noise_plots_brevitas/ber_noise/{model_name}")

    plt.style.use('default')

    test_accs = []

    # Create a CSV file to store the raw data
    csv_file_path = os.path.join(f"noise_plots_brevitas/ber_noise/{model_name}", "raw_data_all_layers.csv")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["BER Value", "Average Accuracy"])

        for ber in ber_vector:
            noisy_models = add_digital_noise_to_model_brevitas(model, layer_names, ber, num_perturbations)

            accuracies = []

            for noisy_model in noisy_models:
                noisy_model.to(device)
                accuracies.append(test(noisy_model, test_quantized_loader, device))

            avg_accuracy = sum(accuracies) / len(accuracies)
            test_accs.append(avg_accuracy)

            # Write the raw data to the CSV file
            writer.writerow([ber, avg_accuracy])

            print("BER Value: {}\tAverage Accuracy: {}".format(ber, avg_accuracy))

    print( "Noise Plot Generated for Multiple Layers - BREVITAS - SAVING" )

    plt.plot(ber_vector, test_accs, label='Accuracy at Different Perturbation Levels for all Layers')

    plt.xlabel('BER Value')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of BER Noise on Test Accuracy (All Layers)')
    plt.legend(title='Layers')
    plt.savefig(f"noise_plots_brevitas/ber_noise/{model_name}/all_layers.png")
    plt.show()
    plt.clf()

    print( "Plot Generated and Saved - END" )

"""
Function Name: gaussian_noise_plots_brevitas()

Parameters:

    1.  num_perturbations:
            Description:        The number of modified models to generate for each value of standard deviation.
            Type:               Integer.
            Details:            Determines how many times the function will create a modified version of the model with Gaussian noise applied for each sigma value.

    2.  layer_names:
            Description:        A list of layer names within the model to which Gaussian noise will be added.
            Type:               List of strings.
            Details:            Each string in this list should correspond to the name of a layer in the model. The function applies Gaussian noise to these specified layers.

    3.  sigma_vector:
            Description:        A list of standard deviation values for the Gaussian noise.
            Type:               List of floats.
            Details:            Each sigma value in this vector represents a different intensity of Gaussian noise to be applied to the model's layers.

    4.  model:
            Description:        The neural network model to be evaluated under different noise conditions.
            Type:               Brevitas layer or similar PyTorch model object.
            Details:            This is the base model on which the Gaussian noise will be applied for testing its performance.

    5.  device:
            Description:        The computational device (e.g., CPU, GPU) where the operation is to be performed.
            Type:               PyTorch device object.
            Details:            Ensures that the model and all operations are performed on the specified device.
 
    6.  test_quantized_loader:
            Description:        DataLoader containing the test dataset for evaluating the model.
            Type:               PyTorch DataLoader object.
            Details:            Used to test the model's performance (accuracy) under various noise conditions.

    7.  analog_noise_type:
            Description:        Flag to determine the type of Gaussian noise addition.
            Type:               Boolean.
            Details:            If True, independent Gaussian noise is added; if False, proportional Gaussian noise is added.

    8.  model_name:
            Description:        Name of the model, used for organizing output files and plots.
            Type:               String.
            Details:            Used to create directories and name files where the results (plots and data) will be saved.

This function gaussian_noise_plots_brevitas adds Gaussian noise to a given model for each layer and creates plots to visualize the effect of noise on test accuracy.

The function first creates a folder to store the noise plots, if it does not already exist. Then, for each layer in layer_names, 
it loops over each value of standard deviation in sigma_vector and adds Gaussian noise to the model for the current layer only. 
It then tests the accuracy of each noisy model and appends the result to a list of test accuracies.

After testing for all standard deviation values, the function plots the test accuracy as a function of the standard deviation 
for the current layer and saves the plot to a file. It then computes the average test accuracy across all layers for each 
standard deviation value and plots the averaged test accuracies as a function of the standard deviation.

Finally, the function saves the average test accuracy plot to a file and displays it.
"""

def gaussian_noise_plots_brevitas(num_perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, analog_noise_type, model_name):

    print( "Generating Gaussian Noise Plot - BREVITAS - START" )

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
                     label='{} Accuracy at Different Perturbation Levels'.format(layer))

        # Compute the average test accuracy across all layers for each standard deviation value
        avg_test_accs = [sum(x) / len(x) for x in zip(*all_test_accs)]

        print( "Gaussian Noise Plot Generated - BREVITAS - SAVING" )

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

        print( "Plot Generated and Saved - END")

"""
Function Name: gaussian_noise_plots_brevitas_all()

Paramters:

    1.  num_perturbations:
            Description:        The number of modified models to generate for each value of standard deviation.
            Type:               Integer.
            Details:            Specifies how many times the function will create a modified version of the model with 
                                Gaussian noise applied for each sigma value.

    2.  layer_names:
            Description:        A list of layer names within the model to which Gaussian noise will be added.
            Type:               List of strings.
            Details:            Each string in this list should correspond to the name of a layer in the model. 
                                The function applies Gaussian noise to these specified layers.

    3.  sigma_vector:
            Description:        A list of standard deviation values for the Gaussian noise.
            Type:               List of floats.
            Details:            Each sigma value in this vector represents a different intensity of Gaussian noise to 
                                be applied to the model's layers.

    4.  model:
            Description:        The neural network model to be evaluated under different noise conditions.
            Type:               Brevitas layer or similar PyTorch model object.
            Details:            This is the base model on which the Gaussian noise will be applied for testing its performance.

    5.  device:
            Description:        The computational device (e.g., CPU, GPU) where the operation is to be performed.
            Type:               PyTorch device object.
            Details:            Ensures that the model and all operations are performed on the specified device.

    6.  test_quantized_loader:
            Description:        DataLoader containing the test dataset for evaluating the model.
            Type:               PyTorch DataLoader object.
            Details:            Used to test the model's performance (accuracy) under various noise conditions.

    7.  analog_noise_type:
            Description:        Flag to determine the type of Gaussian noise addition.
            Type:               Boolean.
            Details:            If True, independent Gaussian noise is added; if False, proportional Gaussian noise is added.

    8.  model_name:
            Description:        Name of the model, used for organizing output files and plots.
            Type:               String.
            Details:            Used to create directories and name files where the results (plots and data) will be saved.
"""

def gaussian_noise_plots_brevitas_all(num_perturbations, layer_names, sigma_vector, model, device, test_quantized_loader, analog_noise_type, model_name):

    print( "Generating Gaussian Noise Plot for Multiple Layers - BREVITAS - START" )

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

        print( "Gaussian Noise Plot for Multiple Layers Generated - BREVITAS - Saving" )

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

        print( "Plot Generated and Saved" )

"""
Function Name: return_noisy_matrix()

Parameters:

    1.  independent
        Description:    Flag to determine the type of noise addition.
        Type:           Boolean.
        Details:        If True, noise is added independently to each element. If False, noise is added in a proportional way to each element.

    dim_matrix
        Description:    Dimensions of the matrix to which noise will be added.
        Type:           Tuple of integers.
        Details:        Specifies the shape of the noise matrix to be generated.

    sigma
        Description:    Standard deviation of the noise.
        Type:           Float.
        Details:        Controls the intensity of the noise in the generated matrix.

    device
        Description:    The computational device (e.g., CPU, GPU) where the operation is to be performed.
        Type:           PyTorch device object.
        Details:        Ensures the generated noisy matrix is allocated on the specified device 
                        (e.g., for compatibility with the rest of a PyTorch model).
"""

def return_noisy_matrix(independent, dim_matrix, sigma, device):

    print( "Generating Noisy Matrix - START" )
    
    if (independent):
        noised_matrix = torch.randn(dim_matrix) * sigma
    else:
        noised_matrix = torch.randn(dim_matrix) * sigma + 1

    print( "Data Received - Matrix Generated - END" )

    return noised_matrix.to(device)
