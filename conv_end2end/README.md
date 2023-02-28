# Convolutional End-2-End Nerual Network

This convolutional end to end neural network is the first one we are going to use to port information to the Pynq-Z1 board.

Full instructions are within the provided notebook but explanations and understanding will be here.

This file must be ran using version `1.22.0` of numpy and will most likely be using a newer version of numpy. To change this version, you must downgrade the numpy package with Anaconda in your terminal. This is done with the following command.

```
conda install -c conda-forge numpy=1.22.0
```

## Table of Contents

1. [Introduction]()
2. [End-to-End Recapitulation]()
3. [Brevitas Export, FINN Import and Tidy Up Transformations]()
4. [Add Pre and Post Processing]()


---


## Introduction

This code is a Jupyter Notebook that demonstrates the FINN (Framework for fast, INterpretable and Neural Network) steps necessary to take a binarized convolutional network all the way down to a heterogeneous streaming dataflow accelerator running on an FPGA. It is recommended to first go through the simpler end-to-end notebook for a fully connected network before proceeding with this one.
CNV-w1a1 Network

The quantized neural network (QNN) targeted in this notebook is CNV-w1a1. This network classifies 32x32 RGB images into one of ten CIFAR-10 classes. All weights and activations in this network are quantized to bipolar values (-1 or +1), with the exception of the input (RGB with 8 bits per channel) and the final output (32-bit numbers). It is composed of a repeating convolution-convolution-maxpool layer pattern to extract features using 3x3 convolution kernels (with weights binarized), followed by fully connected layers acting as the classifier.

## End-to-End Flow Recap

The FINN compiler comes with many transformations that modify the ONNX representation of the network according to certain patterns. This notebook demonstrates a possible sequence of such transformations to take a particular trained network all the way down to hardware. The flow is divided into 5 sections represented by a different color: Brevitas export (green), preparation of the network (blue) for the Vivado HLS synthesis and Vivado IPI stitching (orange), and finally building a PYNQ overlay bitfile and testing it on a PYNQ board (yellow). There is an additional section for functional verification (red) on the left side of the diagram, which is not covered in this notebook.

## Brevitas Export, FINN Import and Tidy-Up

The notebook starts by exporting the pretrained CNV-w1a1 network to ONNX, importing that into FINN, and running the "tidy-up" transformations to have a first look at the topology. Preprocessing and postprocessing steps can be added directly in the ONNX graph. In this case, the preprocessing step divides the input uint8 data by 255 so the inputs to the CNV-w1a1 network are bounded.


## Adding Pre- and Postprocessing

Postprocessing is done using a Softmax activation function to convert the output of the last layer to probabilities. The final ONNX model is generated with preprocessing and postprocessing steps included. 

The model is ready for the next stage of the FINN flow, which involves the use of Vivado HLS and Vivado IP Integrator for generating the hardware design, which is not covered in this notebook.

