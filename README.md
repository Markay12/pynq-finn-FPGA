# pynq-finn-FPGA

FINN and Brevitas Neural Network integration onto Pynq FPGA Board.

[![GitHub Discussions](https://img.shields.io/badge/discussions-join-green)](https://github.com/Xilinx/finn/discussions)
[![ReadTheDocs](https://readthedocs.org/projects/finn/badge/?version=latest&style=plastic)](http://finn.readthedocs.io/)

# Table of Contents

1. About
2. Directory Overview


# Contents

## About

The purpose of this project and repository is to use FINN and Brevitas to run a quantized convolutional neural network (QCNN or CNN) on the Pynq FPGA board.


### Brevitas Overview

Brevitas is the main PyTorch library that is used for network quantization. This Brevitas network has a focus on _quantization-aware-training_ (QAT). 

Software that uses Brevitas is credited with 

```
@software{brevitas,
  author       = {Alessandro Pappalardo},
  title        = {Xilinx/brevitas},
  year         = {2022},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3333552},
  url          = {https://doi.org/10.5281/zenodo.3333552}
}
```

For more information [Brevitas Docs](https://github.com/Xilinx/brevitas)

### FINN Overview
FINN is an experimental framework from the Xilinx Research Labs to explore deep neural network inference on FPGA devices. FINN specifically targets quantized neural networks with emphasis on generating dataflow-style architectures customized for each network. The resulting FPGA accelerators are highly efficient and can yield high throughput and low latency. The framework is fully open-source in order to give a higher degree of flexibility, and is intended to enable neural network research spanning several layers of the software/hardware abstraction stack.

For more information [FINN Github Docs](https://github.com/Xilinx/finn)


### Pynq 

This is the final part of our three main categories to this project.

<img align="right" src="https://raw.githubusercontent.com/Xilinx/finn/github-pages/docs/img/finn-stack.png" alt="drawing" style="margin-right: 20px" width="250"/>

PYNQ is an open-source project from Xilinx that makes it easy to design embedded systems with Zynq All Programmable Systems on Chips (APSoCs). Using the Python language and libraries, designers can exploit the benefits of programmable logic and microprocessors in Zynq to build more capable and exciting embedded systems. PYNQ users can now create high performance embedded applications with

- parallel hardware execution
- high frame-rate video processing
- hardware accelerate algorithms
- real-time signal processing
- high bandwidth IO
- low latency control

Follow for more information [Pynq Docs](http://www.pynq.io/)


### Short Term Goal

The short term goal of this project is to run a simple CNN using Brevitas, Finn and finally deployment to the Pynq board. 

### Long Term Goal

The long term goal of this project is to program this FPGA device while studying the algorithms used and its hardware resilience. This will eventually be used to further empower autonomous navigation with radar sensing and machine learning.

## Directory Overview

1. /docs/

Docs is the location where all setup information will be stored. This includes but is not limited to Pynq information, Ubuntu setup information, FINN and Brevitas information. Each separate setup will have their own subpath and will be updated as time progresses. The setup should be done in chronological order beginning with the Pynq setup information.

 
