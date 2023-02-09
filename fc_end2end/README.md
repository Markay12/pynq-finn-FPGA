# Fully Connected End-to-End Network

This is a fully connected, simple and binarized network trained on the MNIST dataset. What is important about this file and implementation is how it is taken all the way down to a custom bitfile running on the Pynq-Z1 board.

The full process is explained in the Jupyter Notebook but will also be explained here.


Note: Numpy has to be downgraded to 1.22.0 in order to work with this notebook. This cannot be done with just pip uninstall and re-install. This must be done with anaconda. 

To do this with anaconda, execute the below code and answer `y` if prompted.

```
# pip uninstall numpy

# conda install -c conda-forge numpy=1.22.0
```

This should successfully install the version of numpy required for the fully connected notebook completion.


## Table of Contents

1. [Overview]()
	1. [Finn Compiler]()
2. [Outline]()
	1. [Brevitas Export]()

---

## Overview

This is an example of a fully-connected and binarized network trained on the MNIST dataset. Additionally, after the network has been trained we will take it all the way down to a customized bitfile running on the Pynq board.

> Note: This notebook uses Vivado for some editing and can take up to an hour or more to execute. Therefore, watch the time you have wisely.

### FINN Compiler

There are many transformations that modify ONNX representations of networks. This fully connected network works on a possible sequence of transformations to take a trained network all the way down to hardware. This hardware is the Pynq-Z1 board.

<img align="right" src="https://github.com/Markay12/pynq-finn-FPGA/blob/main/Cifar10_Exploration/Brevitas/PynqInHouse2.jpg?raw=true" alt="drawing" style="margin-right: 20px" width="250"/>

While working through the notebook we will use helper functions to show outputs and give you a better understanding of what you are seeing.

1. `showSrc` will show the source code of FINN library calls.
2. `showInNetron` will show the ONNX model at the current transformation step.

Additionally, these netron displays are interactive, but only work when running the notebook actively. Looking at this notebook from github will not show the netron views.



---

## Outline

## Brevitas Export

FINN expects an ONNX mopdel as input which can be trained with Brevitas.

Brevitas is a software library for quantized deep learning in PyTorch. It provides a set of tools and algorithms to perform quantization of neural networks, which can result in smaller models that are quicker are more efficient to run on edge devices (Pynq-Z1). Quantization is a technique to reduce the precision of weights and activations in a neural network to make it possible to represent the network in fewer bits and ultimately perform computation quicker.

The model that is used for this CNN is the TFC-w1a1 model as the example network. These example netwrks can be found on the Xilinx [Github Page](https://github.com/Xilinx/finn/tree/main/notebooks/end2end_example)

Once the model has been exported and loaded with the pretrained weights we can visualize the expoerted model. The exported model will look something like this:

![TFC\_FirstExport](https://github.com/Markay12/pynq-finn-FPGA/blob/main/fc_end2end/assets/TFC_NetronView_1.png?raw=true)


