# Convolutional End-to-End Nerual Network

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
5. [How FINN Implements Convolutions]()
6. [Partitioning, Conversion to HLS Layers and Folding]()

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

## How FINN Implements Convolutions

This section contains the implementation of convolutions in the FINN framework using lowering.  In this approach, convolutions are converted to matrix-matrix multiplication operations by sliding a window over the input image. The resulting dataflow architecture looks similar to a fully connected layer, but with the addition of a sliding window unit that produces the matrix from the input image.

To target this hardware architecture using a Neural Network, a convolution lowering transform, as well as a streamlining is used to get rid of floating point scaling operations. Streamlining simplifies the network along with the lowering transform. The version of streamlining that is used in this implementation is only good for this current network and may not work with other networks that you use.

Along with this, specific transformations are applied to the network using FINN. The Streamline transformation in particular moves floating point scaling and addition operations closer to the input of the nearest thresholding activation and absorbs them into thresholds. 

Additionally, the LowerConvsToMatMul transformation converts ONNX Conv nodes into sequences of Im2Col, MatMul nodes, and the MakeMaxPoolNHWC and AbsorbTransposeIntoMultiThreshold transformations convert the data layout of the network into the NHWC data layout that finn-hlslib primitives use.

Finally, the ConvertBipolarMatMulToXnorPopcount transformation is used to correctly implement bipolar-by-bipolar (w1a1) networks using finn-hlslib. The resulting streamlined and lowered network is visualized using Netron, showing how Conv nodes have been replaced with pairs of Im2Col, MatMul nodes, and many nodes have been replaced with MultiThreshold nodes.

## Partitioning, Conversion to HLS Layers and Folding

The next steps are similar to what was done for the TFC-w1a1 network. Fist we convert the layers that we can put into the FPGA into their HLS equivalents and separate them out into a dataflow partition.

```Python
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_data_layouts import InferDataLayouts

# choose the memory mode for the MVTU units, decoupled or const
mem_mode = "decoupled"

model = ModelWrapper(build_dir + "/end2end_cnv_w1a1_streamlined.onnx")
model = model.transform(to_hls.InferBinaryMatrixVectorActivation(mem_mode))
model = model.transform(to_hls.InferQuantizedMatrixVectorActivation(mem_mode))
# TopK to LabelSelect
model = model.transform(to_hls.InferLabelSelectLayer())
# input quantization (if any) to standalone thresholding
model = model.transform(to_hls.InferThresholdingLayer())
model = model.transform(to_hls.InferConvInpGen())
model = model.transform(to_hls.InferStreamingMaxPool())
# get rid of Reshape(-1, 1) operation between hlslib nodes
model = model.transform(RemoveCNVtoFCFlatten())
# get rid of Tranpose -> Tranpose identity seq
model = model.transform(absorb.AbsorbConsecutiveTransposes())
# infer tensor data layouts
model = model.transform(InferDataLayouts())
parent_model = model.transform(CreateDataflowPartition())
parent_model.save(build_dir + "/end2end_cnv_w1a1_dataflow_parent.onnx")
sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
sdp_node = getCustomOp(sdp_node)
dataflow_model_filename = sdp_node.get_nodeattr("model")
# save the dataflow partition with a different name for easier access
dataflow_model = ModelWrapper(dataflow_model_filename)
dataflow_model.save(build_dir + "/end2end_cnv_w1a1_dataflow_model.onnx")
```

Notice the additional RemoveCNVtoFCFlatten transformation that was not used for TFC-w1a1. In the last Netron visualization, you may have noticed a Reshape operation towards the end of the network where the convolutional part of the network ends and the fully-connected layers started. That Reshape is essentially a tensor flattening operation, which we can remove for the purposes of hardware implementation. We can examine the contents of the dataflow partition with Netron and observe the ConvolutionInputGenerator, MatrixVectorActivation, and StreamingMaxPool\_Batch nodes that implement the sliding window, matrix multiply, and maxpool operations in hlslib. Note that the MatrixVectorActivation instances following the ConvolutionInputGenerator nodes are really implementing the convolutions, despite the name. The final three MatrixVectorActivation instances implement actual FC layers.

```Python
showInNetron(build_dir + "/end2end_cnv_w1a1_dataflow_parent.onnx")
```

Note that pretty much everything has gone into the StreamingDataflowPartition node; the only operation remaining is to apply a Transpose to obtain NHWC input from an NCHW input (the ONNX default).

```python
showInNet
```

