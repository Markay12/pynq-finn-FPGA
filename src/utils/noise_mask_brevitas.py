"""
Noise Mask Utilities
-------------------------

Description:
    This file includes the noise mask utilities script which allows for testing the neural network performance under a mask
    rather than noise. These functions are used in the same manner as the noise_injection_brevitas_funcs, however, they are
    testing a mask rather than independent and / or proportional noise.

Functions:
    add_mask_to_model_brevitas()
    random_clust_mask_brevitas()

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
import numpy as np
from noise_injection_brevitas_funcs import return_noisy_matrix
import torch

"""
Function Name: add_mask_to_model_brevitas()

Parameters: 
    1. model
        Description:        The neural network model where the noise will be applied.
        Type:               Brevitas layer or a PyTorch object.
        Details:            This is the target of the noise injection. The function iterates over specified layers of this model to apply the noise mask.

    2. device
        Description:        The specific device that will be used to add noise to the model. This can either be your GPU device or CPU device. In most cases the
                            GPU will accelerate the added noise time greatly.
        Type:               PyTorch device object.
        Details:            This parameter ensures that all operations are performed on the correct device, aligning with the model's device placement.

    3. layer_names          
        Description:        This is a list of the layer names that are to have their weights perturbed. Each layer name is specific to the model and how the
                            model keys are structured.
        Type:               List of strings.
        Details:            Each string in this list should correspond to the name of a layer in the model. The function applies the noise mask to each of 
                            these layers.

    4. P
        Description:        Target density for the noise mask.
        Type:               Float value.
        Details:            Similar to the P parameter in the random_clust_mask_brevitas function, it represents the fraction of the mask that should be non-zero.

    5. gamma   
        Description:        Controls the falloff rate of the 2D filter in a frequency space.
        Type:               Float value.
        Details:            Influences the distribution of noise in the generated mask, as in the random_clust_mask_brevitas function.

    6. num_perturbations
        Description:        This describes the number of perturbations that will be run though the model. The larger value, the more time the model is accessed and
                            perturbed. This leads to a better set of data for averages and smoother samples.
        Type:               Float value / Integer.
        Details:            This parameter dictates how many times the function will iterate to create different versions of the model with noise masks applied.

    7. sigma
        Description:        Standard deviation for the noise generation.
        Type:               Float value.
        Details:            Used in generating the noise matrix when applying independent noise to the weights.

    8. independent_type
        Description:        This value determines whether the type of noise will be added independently or proportionally.
        Type:               Float value that works as a boolean. 0 for proportional and 1 for independent.
        Details:            Determines the method or distribution to use for generating the noise matrix.

    9. print_weights   
        Description:        This determines whether or not the weights will be printed or not for debugging puposes.
        Type:               Boolean.

This function takes a brevitas layer, a mask generated by the random_clust_mask function,
and two parameters P and gamma. It generates a random clustered mask using the random_clust_mask
function, and then sets the weights at the locations in the mask to zero, effectively sparsifying
the layer's weights.
"""

def add_mask_to_model_brevitas(model, device, layer_names, P, gamma, num_perturbations, sigma, independent_type, print_weights=False):

    modified_models = []

    for _ in range(num_perturbations):

        modified_model = deepcopy(model)

        for layer_name in layer_names:

            layer = getattr(modified_model, layer_name)

            with torch.no_grad():

                # get weights of the tensors
                weight_tensor = layer.weight.clone().detach()
                
                #print(weight_tensor.shape)
                
                # generate mask with correct shape
                mask_graph, mask_numeric, mask_logical = random_clust_mask_brevitas(weight_tensor, P, gamma)

                if print_weights:
                    print("Weights before masking:")
                    print(weight_tensor)
                    
                #plt.imshow(weight_tensor.cpu().numpy(), cmap='viridis', aspect = 'auto')
                #plt.colorbar()
                #plt.show()
                #plt.clf()

                # apply mask to the whole weight tensor
                #print("mask logical = ",mask_logical)
                #print("mask numeric = ",mask_numeric)
                
                #height, width = mask_graph.shape
                #                
                #plt.imshow(mask_graph.cpu().numpy(), cmap='viridis', aspect='auto', extent=[0, width - 1, 0, height - 1])
                #plt.colorbar()
                #plt.title("Mask Graph")
                #plt.show()
                #plt.clf()
                
                if (independent_type):
                    weight_tensor += mask_numeric * return_noisy_matrix(independent_type, weight_tensor.shape, sigma, device)
                else:
                    noisy_mat = return_noisy_matrix(independent_type, weight_tensor.shape, sigma, device)
                    noisy_mat[~mask_logical] =  1
                    weight_tensor *= noisy_mat
                
                if print_weights:
                    print("Weights after masking:")
                    print(weight_tensor)
                    
                #plt.imshow(weight_tensor.cpu().numpy(), cmap='viridis', aspect = 'auto')
                #plt.colorbar()
                #plt.show()
                #plt.clf()

                # create new weight parameter and assign to layer
                noised_weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                
                layer.weight = noised_weight

        modified_models.append(modified_model)

    return modified_models

"""
Function: random_clust_mask_brevitas()

Paramters: 
    1. weight
        Description:    This parameter represents the weights of a neural network layer or a similar structure that you wish to 
                        apply the noise mask to.
        Type:           PyTorch Tensor
        Details:        The shape of weight determines the dimensions N and M of the noise mask. 
                        The function adapts the mask size based on these dimensions to ensure compatibility with the weights tensor. 
                        Typically, this tensor would be a weight matrix of a neural network layer.
    2. P
        Description:    Target density for the noise mask.
        Type:           Float.
        Details:        P represents the fraction of the mask that you want to be non-zero (or True in a boolean context). 
                        This value should be between 0 and 1, where 0 means no noise and 1 means full noise. 
                        The function iteratively adjusts the threshold to ensure that the fraction of non-zero values in the 
                        generated mask is at least P.
    3. gamma
        Description:    Controls the falloff rate of the 2D filter used in frequency space.
        Type:           Float.
        Details:        gamma is a parameter that influences the shape of the frequency response of the filter applied during 
                        the Fast Fourier Transform process. A higher gamma value typically results in a steeper falloff, 
                        affecting how the noise clusters in the spatial domain after the inverse FFT.

The random_clust_mask function generates a random clustered mask of boolean values with a given size N and target density P. 
It first generates a random matrix of values between 0 and 1, performs a two-dimensional Fast Fourier Transform on this matrix, 
and creates a 2D filter in frequency space that varies inversely with frequency over a frequency vector f. The filter's falloff 
rate is controlled by a parameter gamma. The filtered result is then inverse transformed back to the spatial domain, and a 
threshold T is set to the maximum value in the result. Finally, a boolean mask is initialized with the same dimensions as the 
result, and the threshold T is decreased iteratively until the fraction of nonzero values in the mask is greater than or equal 
to the target density P. The function returns the boolean mask as a PyTorch tensor.
"""

def random_clust_mask_brevitas(weight, P, gamma):

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
    while True:
        mask = ifft_result > T

        current_fraction = np.count_nonzero(mask) / (N * N)

        if current_fraction >= P:
            break

        T -= decrement_step

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
    # Create a temporary tensor that repeats the mask tensor along the 3rd dimension
    temp_mask = mask_tensor.unsqueeze(2).repeat(1, 1, weight.shape[2])

    # Copy the temporary mask tensor along the 4th dimension (axis=3) of the mask_final tensor
    for i in range(mask_final.shape[3]):
        mask_final[:, :, :, i] = temp_mask
        
    mask_logical = torch.zeros(weight.shape, dtype=torch.bool,device=weight.device)
    mask_logical = (mask_final==1)
    mask_numeric = mask_final
    
    #mask tensor is 2D and float type
    #mask numeric/final is 4D and float like
    #mask logical is 4D and bool type
    return mask_tensor, mask_numeric, mask_logical