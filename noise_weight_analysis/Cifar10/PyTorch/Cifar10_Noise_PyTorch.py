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

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

# setting random seed for random variable noise each time
random.seed(time.time())

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Target device: " + str(device))

## Load and Normalize Cifar10 Dataset
# The output of the torchvision dataest are PILImage images of range [0, 1]. These are then transformed into Tensors of normalized range being [-1, 1].

# Load CIFAR-10 dataset
train_set = torchvision.datasets.CIFAR10("./data", download=True, transform=
                                         transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.CIFAR10("./data", download=True, train=False, transform=
                                        transforms.Compose([transforms.ToTensor()]))

# Define the train and validation set sizes
train_size = int(len(train_set) * 0.8)
val_size = len(train_set) - train_size

# Split the dataset into train and validation sets
train_data, val_data = random_split(train_set, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1000)

a = next(iter(train_loader))
print(a[0].size())
print(len(train_set))

print("Samples in each set: train = %d, test = %s" % (len(train_set), len(train_loader))) 
print("Shape of one input sample: " +  str(train_set[0][0].shape))

# set batch size
batch_size = 1000

# Create a DataLoader for a training dataset with a batch size of 100
train_quantized_loader = DataLoader(train_set, batch_size=batch_size)
test_quantized_loader = DataLoader(test_set, batch_size=batch_size)

# Initialize a variable to count the number of batches in the train_loader
count = 0

# Iterate through the train_loader
for x,y in train_loader:

  # Print the shape of the input and label tensors for the current batch
  print("Input shape for 1 batch: " + str(x.shape))
  print("Label shape for 1 batch: " + str(y.shape))

  # Increment the count variable
  count += 1

  # Exit the loop after printing the first batch
  if count == 1:
    break

# Determine if we can use the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Target device: " + str(device))


class CIFAR10CNN(nn.Module):

    def __init__(self):
        super(CIFAR10CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=128*4*4, out_features=1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)
        out = self.fc3(out)

        return out



model = CIFAR10CNN().to(device)


# Create an adaptive learning rate 
learning_rate=0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For the adaptive learning rate
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
print(model)

# ## Train and Test
# 
# This code trains a convolutional neural network (CNN) on the Cifar10 dataset using PyTorch. 
# The code uses the torchvision module to download and preprocess the dataset, and defines a CNN with two convolutional layers,
# two max-pooling layers, and three fully connected layers. The code uses the cross-entropy loss function and the 
# Adam optimizer to train the model.
# 
# To run the code, first make sure PyTorch and torchvision are installed on your system.
# Then, run the code in a Python environment. The code should automatically download the Fashion MNIST dataset
#  and train the model for 20 epochs.
# 
# During training, the code prints the loss after every 100 batches. 
# After each epoch, the code evaluates the model on the test set and prints the test accuracy.


# updating training to add adaptive lr

num_epochs = 25
for epoch in range(num_epochs):
    # training phase
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # validation phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    val_accuracy = 100 * correct_val / total_val

    # call scheduler to adjust learning rate
    scheduler.step(val_loss)

    # print loss and accuracy
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%'
          .format(epoch+1, num_epochs, train_loss/len(train_loader), train_accuracy, val_loss/len(val_loader), val_accuracy))



# Get all model Parameters
trained_state_dict = model.state_dict()

# Get individual weights from layers
weight_layer1 = model.state_dict()['layer1.0.weight']

def test(model, test_loader):
# testing phase
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            
            
    accuracy = 100 * correct / total
    return accuracy

# Sigma is weight dependent
# Range is clipped to wmin/wmax

# Add noise to weight and or bias matrices. 
# The weight is a centered normal around the weight values with a sigma difference.

from copy import deepcopy

# Ensure Randomness
random.seed(time.time())


def add_noise(matrix, sigma):

    # State dependent
    # noised_weight = matrix + matrix*np.random.normal(0,scale=sigma)
    
    noised_weight = np.random.normal(loc=matrix, scale=sigma)
    
    return noised_weight

def add_noise_to_model(model, layers, sigma, num_perturbations):
    
    # keep all modified models accuracy so we can get a correct average
    modified_models = []

    for _ in range(num_perturbations):    

        # create individual modified model
        modified_model = deepcopy(model)

        for i in range(len(layers)): 
            layer_name = f'layer{i + 1}'
            layer = getattr(modified_model, layer_name)
    
            # get weight and bias for layer
            weight_temp = layer[0].weight.data.cpu().detach().numpy()
            bias_temp = layer[0].bias.data.cpu().detach().numpy()
    
        
            # add noise to the weight and bias
            noise_w = add_noise(weight_temp, sigma)
            noise_b = add_noise(bias_temp, sigma)

            # place values back into the modified model for analysis
            layer[0].weight.data = torch.from_numpy(noise_w).float()
            layer[0].bias.data = torch.from_numpy(noise_b).float()

        # append this modified model to the list
        modified_models.append(modified_model)

        
    return modified_models


num_perturbations = 1


# Define the standard deviation values to test
sigma_vector = np.linspace(0, 0.2, 11)

# Define the layer to add noise to
layers = [1]

# Loop over each standard deviation value in sigma_vector
for s in range(len(sigma_vector)):


    # Add noise to the model for the defined layer only
    noisy_models = add_noise_to_model(model, layers, sigma_vector[s], num_perturbations)

    accuracies = []
    
    for noisy_model in noisy_models:
        # Move the model back to the target device
        noisy_model.to(device)

        accuracies.append(test(noisy_model, test_quantized_loader))
    
    # calculate the average accuracy and print the result
    avg_accuracy = sum(accuracies) / len(accuracies)

    # print current sd value and the accuracy
    print("Sigma Value: {}, Average Accuracy: {}%".format(sigma_vector[s], avg_accuracy))


# Increase perturbations for analysis
num_perturbations = 20

### Plotting Analysis
# 
# This code is exploring the effect of adding Gaussian noise to a trained Fashion CNN model on its test accuracy. It first initializes the standard deviation values to be tested and loops over each layer of the model. Within each layer loop, the code initializes an empty list to store the test accuracies for that layer, and then loops over each standard deviation value in the sigma vector. For each standard deviation value, the code creates a copy of the original model and adds Gaussian noise to the parameters of the specified layer only. Then, it tests the accuracy of the noisy model on a quantized test dataset and appends the result to the list of test accuracies for that layer. The code then plots the test accuracies as a function of the standard deviation for that layer.
# 
# After looping over all layers and plotting the test accuracies for each layer, the code initializes an empty list to store the averaged test accuracies for each standard deviation value. It then loops over each layer again, adds noise to the specified layer of the model for each standard deviation value, and tests the accuracy of the noisy model. For each standard deviation value, it appends the test accuracy to the corresponding element of the averaged test accuracies list. The code then plots the test accuracies for each layer as a function of the standard deviation on the same graph and plots the averaged test accuracies as a function of the standard deviation on a separate line. Finally, the code adds labels to the plot and displays it. 
#
#  layer1 is the first convolutional layer, followed by batch normalization, ReLU activation, and max pooling
#  layer2 is the second convolutional layer, followed by batch normalization, ReLU activation, and max pooling
#  fc1 is the first fully connected layer
#  drop is the dropout layer
#  fc2 is the second fully connected layer
#  fc3 is the final fully connected layer (output layer)
###

import os
import matplotlib.pyplot as plt

# Create directory for plots
if not os.path.exists("noise_plots_pytorch"):
    os.makedirs("noise_plots_pytorch")

plt.style.use('default')

# Initialize the standard deviation values
sigma_vector = np.linspace(0, 0.05, 21)

all_test_accs = []

layers = [1,2,3,4,5,6]

# Loop over each layer and plot the test accuracy as a function of the standard deviation for that layer
for layer in layers:
    
    # Initialize a list to store the test accuracies for this layer
    test_accs = []
    
    # Iterate over the standard deviation values and add noise to the model for this layer only
    for sigma in sigma_vector:

        noisy_models = add_noise_to_model(model, [layer], sigma, num_perturbations)

        accuracies = []

        # Test accuracy of each noisy model and append the result to the list
        for noisy_model in noisy_models:
            noisy_model.to(device)

            accuracies.append(test(noisy_model, test_quantized_loader))

        # calculate the average and print
        avg_accuracy = sum(accuracies) / len(accuracies)

        test_accs.append(avg_accuracy)

        print("Sigma value: {}, Average Accuracy: {}%".format(sigma, avg_accuracy))

    all_test_accs.append(test_accs)

    # Plot the test accuracies as a function of the standard deviation for this layer
    plt.plot(sigma_vector, test_accs)
    plt.xlabel('Standard Deviation')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of Noise on Test Accuracy (Layer {})'.format(layer))
    plt.savefig("noise_plots_pytorch/layer_{}.png".format(layer))

    plt.show()

    # clear plot
    plt.clf()

    print("Done with plot Layer {}".format(layer))


# New Average, just showing the average
avg_test_accs = [sum(x) / len(x) for x in zip(*all_test_accs)]

# Plot the averaged test accuracies as a function of the standard deviation
plt.plot(sigma_vector, avg_test_accs, label='Average', linewidth=3, linestyle='--', color="black")

plt.xlabel('Standard Deviation')
plt.ylabel('Test Accuracy')
plt.title('Effect of Noise on Test Accuracy (Average)')

plt.legend()
plt.savefig("noise_plots_pytorch/average.png")
plt.show()

plt.clf()

