## Imports
import numpy as np
import torch
import random

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

def random_clust_mask(N, P, gamma):
    
    # Generate random NxN matrix with values between 0 and 1
    matrix = np.random.rand(N, N)
    
    # Compute 2D FFTransform
    fft_result = np.fft.fft2(matrix)
    
    # 1D Frequency Vector with N bins
    f = np.fft.fftfreq(N)
    f[0] = 1e-6
    
    
    # Create a 2D filter in frequency space that varies inversely with freq over f
    # Gamma controls the falloff rate
    filter_2D = 1/(np.sqrt(f[:, None]**2 + f[None, :] ** 2)) ** gamma
    
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
    
    # Repeat until frac of nonzero values in the mask is greather than or equal to P
    while True:
        mask = ifft_result > T
        
        current_fraction = np.count_nonzero(mask) / (N * N)
        
        if current_fraction >= P:
            break
            
        T -= decrement_step

        
        # decrement_step = max(decrement_step * 0.99, 0.001)
    
    # Return tensor
 
    return torch.tensor(mask, dtype=torch.int)


"""
This function takes a brevitas layer, a mask generated by the random_clust_mask function,
and two parameters P and gamma. It generates a random clustered mask using the random_clust_mask
function, and then sets the weights at the locations in the mask to zero, effectively sparsifying
the layer's weights.
"""

def add_mask_to_weights_brevitas(layer, mask, P, gamma):
    
    # Get the size of the last dimension of layers weights
    N = layer.weight.data.size(-1)
    
    # Generate random clustered mask using function above
    mask = random_clust_mask(N, P, gamma)
    
    # Set the weights at the locations in mask to zero. 
    layer.weight.data[mask] = 0


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

def add_digital_noise_to_brevitas_layer(layer, ber):
    # Get the layer's weight tensor
    weight = layer.weight

    # Convert the weight tensor to a numpy array
    weight_np = weight.detach().numpy()

    # add digital noise to the weight tensor using the `add_digital_noise` function
    noisy_weight_np = add_digital_noise(weight_np, ber)

    # Convert the noisy weight tensor back to a PyTorch tensor
    noisy_weight = torch.from_numpy(noisy_weight_np)

    # Update the layer's weight tensor with the noisy weight tensor
    layer.weight = torch.nn.Parameter(noisy_weight)



## Gaussian Noise Section

"""
This function adds Gaussian noise to a given matrix. The sigma parameter controls
the standard deviation of the noise distribution. The function returns the original
matrix with added noise.
"""

def add_gaussian_noise(matrix, sigma):
    noised_weight = np.random.normal(loc=matrix, scale=sigma)
    return noised_weight


"""
The add_gaussian_noise_to_model_brevitas function takes a Brevitas model, a list of layer names to which noise should be added,
a standard deviation sigma of the Gaussian noise, and an integer num_perturbations representing the number of times to apply 
noise to the specified layers.

For each perturbation, the function makes a deep copy of the original model and adds Gaussian noise to the specified layers
weights and biases using the add_noise function. The noisy weights and biases are then used to update the corresponding layer's 
weight and bias tensors.

After each perturbation, the modified model is added to a list, which is then returned once all perturbations have been applied.
"""

def add_gaussian_noise_to_model_brevitas(model, layer_names, sigma, num_perturbations):
    modified_models = []
    for _ in range(num_perturbations):
        modified_model = deepcopy(model)
        # add noise to the modified model
        for layer_name in layer_names:
            layer = getattr(modified_model, layer_name)
            with torch.no_grad():
                # Get the weight and bias tensors
                weight = layer.weight.detach().numpy()
                bias = layer.bias.detach().numpy() if layer.bias is not None else None
                # Add noise to the weight and bias tensors
                noised_weight = add_noise(weight, sigma)
                if bias is not None:
                    noised_bias = add_noise(bias, sigma)
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
