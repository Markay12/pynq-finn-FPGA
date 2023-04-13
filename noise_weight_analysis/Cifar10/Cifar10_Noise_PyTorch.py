## Cifar 10 Neural Network Perturbation Analysis using PyTorch

# Mark Ashinhust
# April 12, 2023

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch imports
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torchvision import datasets, transforms

# imports for randomness and setting seed
import time
import random

# setting random seed for random variable noise each time
random.seed(time.time())

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Load and Normalize Cifar10 Dataset
# The output of the torchvision dataest are PILImage images of range [0, 1]. These are then transformed into Tensors of normalized range being [-1, 1].

# Transformation and Data Augmentation

# resize image
transform_train = transforms.Compose([transforms.Resize(32, 32)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms(RandomAffine(0, shear = 10, scale = (0.8, 1.2)),
				     transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
