# Fully Connected End-to-End Network

This is a fully connected, simple and binarized network trained on the MNIST dataset. What is important about this file and implementation is how it is taken all the way down to a custom bitfile running on the Pynq-Z1 board.

The full process is explained in the Jupyter Notebook but will also be explained here.

## Table of Contents

---

## Overview

This is an example of a fully-connected and binarized network trained on the MNIST dataset. Additionally, after the network has been trained we will take it all the way down to a customized bitfile running on the Pynq board.

> Note: This notebook uses Vivado for some editing and can take up to an hour or more to execute. Therefore, watch the time you have wisely.

### FINN Compiler

There are many transformations that modify ONNX representations of networks. This fully connected network works on a possible sequence of transformations to take a trained network all the way down to hardware. This hardware is the Pynq-Z1 board.

<img align="right" src="https://github.com/Markay12/pynq-finn-FPGA/blob/main/Cifar10_Exploration/Brevitas/PynqInHouse2.jpg?raw=true" alt="drawing" style="margin-right: 20px" width="250"/>
