"""
Noise Injection Utilities
-------------------------

Description:
    HERE.

Functions:
    add_digital_noise()
    add_gaussian_noise_to_model_pytorch()
    add_gaussian_noise_independent()
    add_gaussian_noise_proportional()

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
    30 December 2024

"""

from copy import deepcopy
import torch

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

    # Generate a new seed each time to add noise to the model.
    # This creates more accurate randomess and noise generation.
    current_time = datetime.datetime.now()
    seed = int( current_time.timestamp() )

    # Set the seed 
    random.seed( seed )

    # Generate a boolean mask with the same shape as the matrix where each element is True with probability `ber`
    mask = np.random.rand( *matrix.shape ) < ber
    
    # Generate a matrix of random values between -1 and 1 with the same shape as the input matrix
    noise = np.random.uniform( -1, 1, size=matrix.shape )
    
    # Add the noise to the input matrix only where the mask is True
    noisy_matrix = matrix + mask * noise
    
    # Clip the noisy matrix to the range [-1, 1] to ensure that all elements are within this range
    return np.clip( noisy_matrix, -1, 1 )

"""
Function Name: add_digital_noise_to_model_pytorch()

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

def add_digital_noise_to_model_pytorch( model, layer_names, ber, num_perturbations ):
    
    modified_models = []
    
    for _ in range(num_perturbations):
        modified_model = deepcopy(model)
        
        for layer_name in layer_names:
            layer = getattr(modified_model, layer_name)

            if layer_name.lower().startswith("c"):
                weight_tensor = layer.conv.weight.detach().clone()
            else:
                weight_tensor = layer.weight.detach().clone()

            # Apply digital noise using PyTorch operations
            noisy_weight = add_digital_noise(weight_tensor, ber)

            # Update weights
            with torch.no_grad():
                if layer_name.lower().startswith("c"):
                    layer.conv.weight = torch.nn.Parameter(noisy_weight)
                else:
                    layer.weight = torch.nn.Parameter(noisy_weight)
        
        modified_models.append(modified_model)
        
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
    
    # keep all modified models accuracy so we can get a correct average
    modified_models = []

    for _ in range( num_perturbations ):    

        # create individual modified model
        modified_model = deepcopy( model )

        for i in range( len( layers ) ): 
            layer_name = f'layer{i + 1}'
            layer = getattr( modified_model, layer_name )
    
            # get weight and bias for layer
            weight_temp = layer[ 0 ].weight.data.cpu().detach().numpy()
            bias_temp = layer[ 0 ].bias.data.cpu().detach().numpy()
    
            # add noise to the weight and bias
            # need to work on this to add independent vs. proportional noise
            noise_w = ( weight_temp, sigma )
            noise_b = ( bias_temp, sigma )

            # place values back into the modified model for analysis
            layer[ 0 ].weight.data = torch.from_numpy( noise_w ).float()
            layer[ 0 ].bias.data = torch.from_numpy( noise_b ).float()

        # append this modified model to the list
        modified_models.append( modified_model )

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

    # Generate a new seed each time to add noise to the model.
    # This creates more accurate randomess and noise generation.
    current_time = datetime.datetime.now()
    seed = int( current_time.timestamp() )

    # Set the seed 
    random.seed( seed )

    noised_matrix = matrix + np.random.normal( 0, scale=sigma )

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

    # Generate a new seed each time to add noise to the model.
    # This creates more accurate randomess and noise generation.
    current_time = datetime.datetime.now()
    seed = int( current_time.timestamp() )

    # Set the seed 
    random.seed( seed )

    noised_matrix = matrix * np.random.normal( 1, scale=sigma )

    return noised_matrix