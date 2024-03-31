"""
Models Setup 
-------------------------

Description:
    Description here.

Classes:
    CIFAR10CNN()
    CIFAR100CNN_4BLOCKS()
    CIFAR100CNN_5BLOCKS()
    CommonUintActQuant()
    CommonIntWeightPerTensorQuant()
    ConvBlock()
    
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
    05 December 2023

Last Modified:
    31 April 2024

"""

# Brevitas imports
import brevitas.nn as qnn
from brevitas.quant import Int32Bias, Int8WeightPerTensorFloat, Uint8ActPerTensorFloatMaxInit
from brevitas.core.restrict_val import RestrictValueType
import torch.nn.functional as F

import torch
import torch.nn as nn

"""
CIFAR10CNN: A Quantized Convolutional Neural Network for CIFAR-10 Dataset Classification

This class implements a convolutional neural network (CNN) designed specifically for the CIFAR-10 image dataset classification task.
It employs quantized neural network layers from the 'torch.nn' and 'brevitas.nn' (as qnn) libraries to efficiently handle computations
with lower bit-width representations. The use of quantization aims to reduce the model size and computational complexity, making it
suitable for deployment in resource-constrained environments such as embedded systems or IoT devices.

The network architecture comprises a series of quantized convolutional layers (QuantConv2d) interspersed with quantized ReLU activation
functions (QuantReLU). It includes max-pooling after certain convolutional layers to reduce spatial dimensions. The final part of the
network flattens the output of the convolutional layers and passes it through quantized fully connected layers (QuantLinear) for
classification into 10 CIFAR-10 classes.

Key components:
- QuantIdentity: Initial quantization layer to quantize the input images into lower bit-width representations.
- QuantConv2d: Quantized convolutional layers with specified bit-width for weights and bias, utilizing Int32Bias for bias quantization.
- QuantReLU: Quantized ReLU activation functions to introduce non-linearity, with specified bit-width for the activations.
- QuantLinear: Quantized fully connected (linear
"""

class CIFAR10CNN(nn.Module):

    def __init__(self):
        super( CIFAR10CNN, self ).__init__()
        self.quant_inp = qnn.QuantIdentity( bit_width=4, return_quant_tensor=True )

        self.layer1 = qnn.QuantConv2d( 3, 32, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias )
        self.relu1 = qnn.QuantReLU( bit_width=4, return_quant_tensor=True )

        self.layer2 = qnn.QuantConv2d( 32, 32, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias )
        self.relu2 = qnn.QuantReLU( bit_width=4, return_quant_tensor=True )

        self.layer3 = qnn.QuantConv2d( 32, 64, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias )
        self.relu3 = qnn.QuantReLU( bit_width=4, return_quant_tensor=True )

        self.layer4 = qnn.QuantConv2d( 64, 64, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias )
        self.relu4 = qnn.QuantReLU( bit_width=4, return_quant_tensor=True )

        self.layer5 = qnn.QuantConv2d( 64, 64, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias )
        self.relu5 = qnn.QuantReLU( bit_width=4, return_quant_tensor=True )

        self.fc1 = qnn.QuantLinear( 64 * 8 * 8, 512, bias=True, weight_bit_width=4, bias_quant=Int32Bias )
        self.relu6 = qnn.QuantReLU( bit_width=4, return_quant_tensor=True )

        self.fc2 = qnn.QuantLinear( 512, 10, bias=True, weight_bit_width=4, bias_quant=Int32Bias )

    def forward( self, x ):
        x = self.quant_inp( x )
        x = self.relu1( self.layer1( x ) )
        x = self.relu2( self.layer2( x ) )
        x = F.max_pool2d( x, 2 )

        x = self.relu3( self.layer3( x ) )
        x = self.relu4( self.layer4( x ) )
        x = F.max_pool2d( x, 2 )

        x = self.relu5( self.layer5( x ) )

        x = x.view( x.size( 0 ), -1 )

        x = self.relu6( self.fc1( x ) )
        x = self.fc2( x )

        return x

"""
CIFAR100CNN_4Blocks: A Quantized Convolutional Neural Network for CIFAR-100 Dataset

This class defines a convolutional neural network (CNN) architecture optimized for classifying images from the CIFAR-100 dataset.
The network is composed of four main blocks, each containing two convolutional layers encapsulated in a 'ConvBlock' structure. This
modular design facilitates ease of adjustments to layer parameters and promotes readability. The network employs quantization both
for convolutional and fully connected layers to ensure efficiency in computational resource usage, making it suitable for deployment
in environments with limited computational capacity.

The architecture is designed to progressively increase the complexity and depth of features extracted from the input images, moving
from simpler patterns in the initial layers to more complex features in the deeper layers. This is achieved by varying the bit width
of the weights and activations across the blocks, with an option to customize these parameters for each block during initialization.

Parameters:
- layer_one_bw (int): Bit width for weights in the first block.
- layer_two_bw (int): Bit width for weights in the second block.
- layer_three_bw (int): Bit width for weights in the third block.
- layer_four_bw (int): Bit width for weights in the fourth block.
- fcn_bw (int): Bit width for weights in the fully connected layers.
- act_bits (int): Bit width for activations throughout the network.
- num_classes (int): Number of classes to predict. Defaults to 100 for CIFAR-100.
- normalization_func (str): Type of normalization function to use ('batch' or other types if implemented).
- dropout (float): Dropout rate for regularization.

Key components:
- ConvBlock: Custom convolutional block that includes convolutional layers, activation functions, optional normalization, and dropout.
  Parameters such as bit width for weights and activations can be adjusted per block.
- qnn.QuantLinear: Quantized fully connected layers that map the features extracted by the convolutional blocks to the output classes.
- qnn.QuantReLU: Quantized ReLU activation function for introducing non-linearity between the fully connected layers.

The model utilizes striding in certain ConvBlocks to reduce the spatial dimensions of feature maps, thereby reducing computational
requirements and focusing on the most relevant features for classification.

Usage:
This model can be instantiated with specific bit widths for each convolutional block and the fully connected layers, allowing for
customization according to precision requirements or computational constraints. It is particularly suited for applications where
model efficiency is paramount, such as edge computing or mobile deployment.
"""


class CIFAR100CNN_4Blocks(nn.Module):
    def __init__(self,layer_one_bw=16,layer_two_bw=16,layer_three_bw=16,layer_four_bw=16,fcn_bw=16,act_bits=16,num_classes=100,normalization_func='batch',dropout=0.1):
        super(CIFAR100CNN_4Blocks, self).__init__()

        # Block 1
        self.conv1 = ConvBlock(3,64,3,weight_bit_width=layer_one_bw, act_bit_width=act_bits,stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv2 = ConvBlock(64,64,3,weight_bit_width=layer_one_bw, act_bit_width=act_bits,stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)

        # Block 2
        self.conv3 = ConvBlock(64,128,3,weight_bit_width=layer_two_bw, act_bit_width=act_bits,stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv4 = ConvBlock(128,128,3,weight_bit_width=layer_two_bw, act_bit_width=act_bits,stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)
        

        # Block 3
        self.conv5 = ConvBlock(128,128,3,weight_bit_width=layer_three_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv6 = ConvBlock(128,128,3,weight_bit_width=layer_three_bw, act_bit_width=act_bits,stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)
        

        # Block 4
        self.conv7 = ConvBlock(128,256,3,weight_bit_width=layer_four_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv8 = ConvBlock(256,256,3,weight_bit_width=layer_four_bw,act_bit_width=act_bits, stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)
        
        # Flatten then FCN
        self.fc1 = qnn.QuantLinear(256, 256, weight_bit_width=fcn_bw,bias=False)
        self.relu1 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bits,
            scaling_per_channel=False,
            return_quant_tensor=True
            )

        self.fc2 = qnn.QuantLinear(256, 100, weight_bit_width=fcn_bw,bias=False)

    def forward(self, x):
        # Unit 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        # Unit 2
        x = self.conv3(x)
        x = self.conv4(x)
        #x = F.max_pool2d(x, 2)

        # Unit 3
        x = self.conv5(x)
        x = self.conv6(x)
        #x = F.max_pool2d(x, 2)

        # Unit 4
        x = self.conv7(x)
        x = self.conv8(x)
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x
    
"""
CIFAR100CNN_5Blocks: An Advanced Quantized Convolutional Neural Network for CIFAR-100 Dataset Classification

This class implements an enhanced convolutional neural network (CNN) architecture tailored for high-accuracy classification on the CIFAR-100 dataset.
Expanding on simpler models, this version introduces a fifth convolutional block, aiming to capture more complex and nuanced features within the dataset.
The network is quantized, employing low bit-width computations for both weights and activations across its layers, making it particularly suitable
for deployment in scenarios where computational resources are limited, such as edge devices or mobile platforms.

The architecture is structured into five sequential blocks of convolutional layers, with each block designed to progressively increase in complexity
and depth of the extracted features. This approach allows the model to effectively learn from the rich and varied images within the CIFAR-100 dataset.
The model also incorporates quantized fully connected layers towards the end, which serve to classify the extracted features into 100 distinct classes.

Initialization Parameters:
- layer_one_bw (int): Bit width for weights in the first convolutional block.
- layer_two_bw (int): Bit width for weights in the second convolutional block.
- layer_three_bw (int): Bit width for weights in the third convolutional block.
- layer_four_bw (int): Bit width for weights in the fourth convolutional block.
- layer_five_bw (int): Bit width for weights in the fifth convolutional block.
- fcn_bw (int): Bit width for weights in the fully connected layers.
- act_bits (int): Bit width for activations throughout the network.
- num_classes (int): Number of classification classes. Set to 100 for CIFAR-100.
- normalization_func (str): Specifies the type of normalization function to be used within the convolutional blocks ('batch' by default).
- dropout (float): Dropout rate applied within the convolutional blocks for regularization.

Model Architecture Overview:
- ConvBlock: Customizable convolutional blocks consisting of convolutional layers followed by activation functions, optionally including normalization and dropout. The bit width for weights and activations can be adjusted per block.
- qnn.QuantLinear: Quantized fully connected layers, mapping the deep feature representations to class predictions. These layers are configured without bias and use specified bit width for weights.
- qnn.QuantReLU: Quantized ReLU activation function employed after the first fully connected layer, facilitating non-linear transformations of the learned features.

The forward pass of the model organizes the processing flow through the convolutional blocks, applying max-pooling at strategic points to reduce feature dimensions while retaining essential information. The network concludes with a flattening step followed by the quantized fully connected layers to produce the final class predictions.

Usage:
This model is ideal for research or applications requiring efficient and high-performing image classification capabilities on the CIFAR-100 dataset, particularly when computational efficiency is a consideration. The customizable bit width parameters for both convolutional and fully connected layers allow for fine-tuning the balance between accuracy and computational resource requirements.
"""


class CIFAR100CNN_5Blocks(nn.Module):
    def __init__(self,layer_one_bw=16,layer_two_bw=16,layer_three_bw=16,layer_four_bw=16,layer_five_bw=16,fcn_bw=16,act_bits=16,num_classes=100,normalization_func='batch',dropout=0.1):
        super(CIFAR100CNN_5Blocks, self).__init__()

        # Block 1
        self.conv1 = ConvBlock(3,64,3,weight_bit_width=layer_one_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv2 = ConvBlock(64,64,3,weight_bit_width=layer_one_bw, act_bit_width=act_bits,stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)

        # Block 2
        self.conv3 = ConvBlock(64,128,3,weight_bit_width=layer_two_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv4 = ConvBlock(128,128,3,weight_bit_width=layer_two_bw,act_bit_width=act_bits, stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)
        
   
        # Block 3
        self.conv5 = ConvBlock(128,128,3,weight_bit_width=layer_three_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv6 = ConvBlock(128,128,3,weight_bit_width=layer_three_bw,act_bit_width=act_bits, stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)
        

        # Block 4
        self.conv7 = ConvBlock(128,256,3,weight_bit_width=layer_four_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv8 = ConvBlock(256,256,3,weight_bit_width=layer_four_bw,act_bit_width=act_bits, stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)

        # Block 5
        self.conv9 = ConvBlock(256,256,3,weight_bit_width=layer_five_bw,act_bit_width=act_bits, stride = 1, padding=1, normalization_func=normalization_func, dropout=dropout)
        self.conv10 = ConvBlock(256,256,3,weight_bit_width=layer_five_bw,act_bit_width=act_bits, stride = 2, padding=1, normalization_func=normalization_func, dropout=dropout)
        
        # Flatten then FCN
        self.fc1 = qnn.QuantLinear(256, 256, weight_bit_width=fcn_bw,bias=False)
        self.relu1 = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bits,
            scaling_per_channel=False,
            return_quant_tensor=True
            )

        self.fc2 = qnn.QuantLinear(256, 100, weight_bit_width=fcn_bw,bias=False)

    def forward(self, x):
        # Organize Image
        # Unit 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        # Unit 2
        x = self.conv3(x)
        x = self.conv4(x)
        #x = F.max_pool2d(x, 2)

        # Unit 3
        x = self.conv5(x)
        x = self.conv6(x)
        #x = F.max_pool2d(x, 2)

        # Unit 4
        x = self.conv7(x)
        x = self.conv8(x)
        x = F.max_pool2d(x, 2)

        # Unit 5 
        x = self.conv9(x)
        x = self.conv10(x)

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x

"""
CommonUintActQuant: A Universal Unsigned Activation Quantizer for Per-Tensor Quantization

This class defines a common unsigned activation quantizer based on Uint8ActPerTensorFloatMaxInit. It's tailored to handle
activation quantization across various neural network layers, where the bit width is determined by the specific layer using
this quantizer. The quantizer focuses on per-tensor quantization, ensuring that each tensor's quantization parameters are
uniquely determined.

Attributes:
- scaling_min_val (float): The minimum scaling value to prevent division by zero during the scaling calculation. Set to a very
  small value by default.
- bit_width (int): The bit width for the quantization. This is intentionally set to None to force specification at the layer level.
- max_val (float): The maximum value for quantization range, aiming to constrain the activations within a specified range.
- restrict_scaling_type (RestrictValueType): Defines the restriction type on the scaling factor, set to logarithmic fixed-point representation
  for precise control over quantization scaling.

Usage:
This quantizer is utilized within quantized activation layers, where the bit width and other parameters are explicitly defined per layer,
allowing for flexible and efficient quantization schemes tailored to the specific needs of each layer within the network.
"""


class CommonUintActQuant(Uint8ActPerTensorFloatMaxInit):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    max_val = 6.0
    restrict_scaling_type = RestrictValueType.LOG_FP

"""
CommonIntWeightPerTensorQuant: A Common Per-Tensor Weight Quantizer

This class implements a generic weight quantizer based on Int8WeightPerTensorFloat, designed to apply per-tensor quantization
for neural network weights. Similar to the activation quantizer, the bit width for the quantization process is not predefined
and must be specified by each implementing layer. This design choice facilitates custom quantization schemes across different
layers of a neural network.

Attributes:
- scaling_min_val (float): Minimum scaling value, avoiding division by zero in the scaling computation. It's set to a very low
  value by default.
- bit_width (int): Bit width for quantization. This attribute is set to None to ensure that the bit width is explicitly defined
  at the layer level.

Usage:
This quantizer is intended for use in quantized convolutional or fully connected layers where weight quantization is necessary.
By not predefining the bit width, it allows each layer to customize its quantization scheme, thereby optimizing the network's
performance and efficiency according to its specific requirements.
"""


class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
    """
    Common per-tensor weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None

"""
ConvBlock: A Customizable Convolutional Block for Quantized Neural Networks

This modular convolutional block integrates quantized convolution, normalization, dropout, and quantized activation into a single
reusable unit. It is designed to be a fundamental building block for constructing complex quantized neural network architectures, 
offering a high degree of customization in terms of quantization parameters and layer configurations.

Parameters:
- in_channels (int): Number of input channels.
- out_channels (int): Number of output channels.
- kernel_size (int or tuple): Size of the convolving kernel.
- weight_bit_width (int): Bit width for weight quantization.
- act_bit_width (int): Bit width for activation quantization, defaulting to 16.
- stride (int or tuple): Stride of the convolution. Defaults to 1.
- padding (int or tuple): Zero-padding added to both sides of the input. Defaults to 0.
- groups (int): Number of blocked connections from input channels to output channels. Defaults to 1.
- bn_eps (float): Epsilon value for batch normalization to avoid division by zero. Defaults to 1e-5.
- activation_scaling_per_channel (bool): Whether to scale activations per channel. Defaults to False.
- normalization_func (str): Type of normalization function ('batch', 'instance', or 'none').
- dropout (float): Dropout rate. Defaults to 0.0.
- return_quant_tensor (bool): Whether the output tensor should be a quantized tensor.

The block sequentially applies a quantized convolutional layer, normalization (BatchNorm2d or InstanceNorm2d, or no normalization),
dropout, and a quantized ReLU activation. It supports flexibility in specifying the bit width for both weights and activations,
allowing for tailored quantization schemes that can optimize the trade-off between accuracy and computational efficiency.

Usage:
ConvBlock serves as a versatile component for constructing quantized CNN architectures, suitable for a wide range of tasks
from image classification to more complex pattern recognition challenges in quantized settings.
"""


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            weight_bit_width,
            act_bit_width=16,
            stride=1,
            padding=0,
            groups=1,
            bn_eps=1e-5,
            activation_scaling_per_channel=False,
            normalization_func='batch',
            dropout=0.0,
            return_quant_tensor=True ):
        super(ConvBlock, self).__init__()
        self.conv = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            weight_quant=CommonIntWeightPerTensorQuant,
            weight_bit_width=weight_bit_width)

        if normalization_func == 'batch':
            self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        elif normalization_func == 'instance':
            self.bn = nn.InstanceNorm2d(num_features=out_channels, eps=bn_eps)
        elif normalization_func == 'none':
            self.bn = lambda x: x
        else:
            raise NotImplementedError(f'{normalization_func} is not a supported normalization function')
        self.dropout = nn.Dropout2d(p=dropout)
        self.activation = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            scaling_per_channel=False,
            return_quant_tensor=return_quant_tensor
            )
            
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x  
    
