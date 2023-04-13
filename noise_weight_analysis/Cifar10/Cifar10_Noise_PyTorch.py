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

import os

# setting random seed for random variable noise each time
random.seed(time.time())

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Target device: " + str(device))

## Load and Normalize Cifar10 Dataset
# The output of the torchvision dataest are PILImage images of range [0, 1]. These are then transformed into Tensors of normalized range being [-1, 1].

# Transformation and Data Augmentation

transform_train = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
                               ])

transform = transforms.Compose([transforms.Resize((32,32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

# Training dataset and validation
training_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
 
# Batch size of 100 i.e to work with 100 images at a time
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True) 
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 100, shuffle=False)

# Convert the images to numpy arrays
def im_convert(tensor):  
  image = tensor.cpu().clone().detach().numpy() # This process will happen in normal cpu.
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image


# Different classes in Cifar10 dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

