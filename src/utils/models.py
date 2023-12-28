# Brevitas imports
import brevitas.nn as qnn
from brevitas.quant import Int32Bias, Int8WeightPerTensorFloat, Uint8ActPerTensorFloatMaxInit
from brevitas.core.restrict_val import RestrictValueType
import torch.nn.functional as F

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