# MLP Perturbed Weight and Bias Analysis

The purpose of this code is to get a general way to add noise to the weights and biases of a MLP network in Python. Upon adding noise, we compare this against the base network to see how it was affected.

This Jupyter notebook demonstrates everything from how to create, train and export a quantized Multi Layer Perceptron (MLP) with quantized weights and activations using Brevitas.

The dataset used for training is a variant of the UNSW-NB15 dataset, which is a network-based dataset that reflects modern network traffic scenarios.
The MLP is small enough to train on a modern x86 CPU, and no GPU is required to train it. Alternatively, pre-trained parameters are provided for the MLP if you want to skip the training process.

The notebook contains the following sections:

1. Load the UNSW_NB15 Dataset
2. Define a PyTorch device
3. Define the Quantized MLP Model
4. Define the Test Method
5. Load with Pre-Trained Parameters
6. Test with Pre-Trained Parameters
7. Add Noise to the Weights and Biases
8. Network Surgery Before Export
9. Export to FINN-ONNX


The binarized representation of the dataset is created by following the procedure defined by Murovic and Trost, and the dataset is wrapped as a PyTorch TensorDataset. The dataset is then wrapped in a PyTorch DataLoader for easier access in batches. The code is written in Python and uses PyTorch and Brevitas libraries.

The repository provides a pre-quantized version of the dataset for convenience. Downloading the original dataset and quantizing it can take some time. The README file in the repository contains instructions to download the dataset.

The repository can be used as a tutorial to learn how to train a quantized neural network (QNN) and how to deploy it as an FPGA accelerator generated by the FINN compiler.

A Quantized MLP model is defined using the Brevitas library,  which trains a multi-layer perceptron with 2-bit weights and activations. The MLP has four fully connected layers, with three hidden layers containing 64 neurons each and a final output layer with a single output. The input size is 593, and the number of classes is one. The network is trained using a DataLoader, which feeds the model with a new predefined batch of training data in each iteration, until the entire training data is fed to the model. Each repetition of this process is called an epoch. The model is tested using a test_loader, and the accuracy is calculated using the accuracy_score method from the sklearn.metrics library.

The code also provides two options for training the network: training from scratch or using pre-trained parameters. The pre-trained parameters should achieve ~91.9% test accuracy. Network surgery is performed before export, where noise is added to the weight and/or bias matrices to make the model more robust. The effect of noise on test accuracy is plotted, and the averaged test accuracies are calculated for each standard deviation value.