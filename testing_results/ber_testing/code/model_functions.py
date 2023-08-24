"""
Description: Quantized Neural Network Models for CIFAR-100 Classification

This module implements quantized convolutional neural networks for the CIFAR-100 image classification task using the Brevitas library.
Brevitas is an extension of the PyTorch deep learning platform that supports quantized neural networks.

Key Features:

    Custom Quantization Layers: The module utilizes custom quantization configurations including CommonUintActQuant for unsigned activations
                                and CommonIntWeightPerTensorQuant for weights.

    Convolutional Blocks: The ConvBlock class is a modular unit that constructs a quantized convolution layer followed by batch normalization
                          (or instance normalization) and a quantized ReLU activation. Dropout can also be included for regularization.

    Models:
        CIFAR100CNN_4Blocks: A quantized CNN model constructed with four ConvBlock units followed by fully connected layers.
                             Pooling layers are strategically placed after certain convolutional units.
        CIFAR100CNN_5Blocks: An extended version of the previous model with five ConvBlock units.

    Flexibility:
        Bit-width Control: Users can specify the bit-width for weights and activations for different layers, enabling experimentation with mixed-precision quantization.
        Normalization Option: Users can choose between batch normalization, instance normalization, or no normalization.
        Dropout: An optional dropout can be added for each ConvBlock for regularization.

Usage:
To use this module, simply instantiate one of the models (CIFAR100CNN_4Blocks or CIFAR100CNN_5Blocks), specifying the bit-width for weights and activations as desired.
Pass an input tensor through the model to obtain the CIFAR-100 classification results.

Dependencies:

    Brevitas: Used for all quantized layers including quantized convolutions, ReLU, and linear layers.
    PyTorch: Used for defining the deep learning model structure, including non-quantized batch normalization, instance normalization, and dropout.


Date: 24 August 2023

"""

#############################
#####      Imports      #####
#############################

# Brevitas Imports
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int32Bias, Int8ActPerTensorFloat, Int8Bias, Uint8ActPerTensorFloatMaxInit, Int8ActPerTensorFloatMinMaxInit
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
import torch.nn.functional as F

# Torch imports
import torch
import torch.nn as nn


class CommonUintActQuant(Uint8ActPerTensorFloatMaxInit):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    max_val = 6.0
    restrict_scaling_type = RestrictValueType.LOG_FP


class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
    """
    Common per-tensor weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None


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
