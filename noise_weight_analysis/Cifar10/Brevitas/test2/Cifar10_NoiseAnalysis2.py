


from copy import deepcopy
import os
from unicodedata import decimal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

# add imports for randomness
import time
import random

import sys

# Brevitas imports
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int32Bias
import torch.nn.functional as F

# For adaptive learning rate import
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split


## Imports from utils file for my defined noise functions
import sys
sys.path.append('C:/Users/ashin/source/repos/Cifar10_Pytorch_NoiseAnalysis/Cifar10_Pytorch_NoiseAnalysis/pynq-finn-FPGA/noise_weight_analysis/utils/')

from noise_functions import random_clust_mask, add_mask_to_model_brevitas, mask_noise_plots_brevitas, add_digital_noise, add_digital_noise_to_model_brevitas, ber_noise_plot_brevitas, add_gaussian_noise, add_gaussian_noise_to_model_brevitas, gaussian_noise_plots_brevitas, test 





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Target device: " + str(device))




from torchvision import transforms

# Define data augmentation transforms
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Apply data augmentation to the training dataset
train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)

# Use the validation transform for the validation dataset
val_set =torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=val_transform)

# Create the data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=4)


a = next(iter(train_loader))
print(a[0].size())
print(len(train_set))

print("Samples in each set: train = %d, test = %s" % (len(train_set), len(train_loader))) 
print("Shape of one input sample: " +  str(train_set[0][0].shape))

## Data Loader
#
# Using PyTorch dataloader we can create a convenient iterator over the dataset that returns batches of data, rather than requiring manual batch creation.

# set batch size
batch_size = 1000

# Create a DataLoader for a training dataset with a batch size of 1000
train_quantized_loader = DataLoader(train_set, batch_size=batch_size)
test_quantized_loader = DataLoader(val_set, batch_size=batch_size)





count = 0

print("\nDataset Shape:\n-------------------------")
for x, y in train_loader:
    print("Input shape for 1 batch: " + str(x.shape))
    print("Label shape for 1 batch: " + str(y.shape))
    count += 1
    if count == 1:
        break




class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=4, return_quant_tensor=True)

        self.layer1 = qnn.QuantConv2d(3, 32, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu1 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer2 = qnn.QuantConv2d(32, 32, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer3 = qnn.QuantConv2d(32, 64, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu3 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer4 = qnn.QuantConv2d(64, 64, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu4 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.layer5 = qnn.QuantConv2d(64, 64, 3, padding=1, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu5 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.fc1 = qnn.QuantLinear(64 * 8 * 8, 512, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu6 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        self.fc2 = qnn.QuantLinear(512, 10, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = F.max_pool2d(x, 2)

        x = self.relu3(self.layer3(x))
        x = self.relu4(self.layer4(x))
        x = F.max_pool2d(x, 2)

        x = self.relu5(self.layer5(x))

        x = x.view(x.size(0), -1)

        x = self.relu6(self.fc1(x))
        x = self.fc2(x)

        return x



# Import testing
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import precision_recall_fscore_support

# Initialize the model, optimizer, and criterion
model = CIFAR10CNN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
criterion = nn.CrossEntropyLoss()

num_epochs = 80
best_test_accuracy = 0
patience = 8
no_improvement_counter = 0


for epoch in range(num_epochs):

    # training phase
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
    
    # Initialize the validation loss
    val_loss = 0
    
    # testing phase
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()  # accumulate the validation loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
        val_loss /= len(val_loader)  # calculate the average validation loss
    
        # Update the learning rate using the validation loss
        scheduler.step(val_loss)

        test_accuracy = 100 * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
            
        if no_improvement_counter >= patience:
            print("Early stopping")
            break

        print('Epoch [{}/{}], Test Accuracy: {:.2f}%, Precision: {:.2f}, Recall: {:.2f}, F1 score: {:.2f}'.format(epoch+1, num_epochs, test_accuracy, precision, recall, f1))



# Print the best test accuracy
print("The final best test accuracy is: {:.2f}%".format(best_test_accuracy))




trained_state_dict = model.state_dict()


# ## Saving Model



if test_accuracy > best_test_accuracy:
    best_test_accuracy = test_accuracy
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_test_accuracy': best_test_accuracy,
        'epoch': epoch
        }, 'best_model.pth')
    no_improvement_counter = 0




# Create an instance of your neural network model
model = CIFAR10CNN().to(device)

# Load the saved state dictionary from file
state_dict = torch.load('best_model.pth')

# Load the state dictionary into the model
model.load_state_dict(state_dict)


def ber_noise_plot_brevitas(num_perturbations, layer_names, ber_vector, model, device, test_quantized_loader):
    
    if not os.path.exists("noise_plots_brevitas/ber_noise/"):
        os.makedirs("noise_plots_brevitas/ber_noise/")
    
    plt.style.use('default')
    
    all_test_accs = []

    for layer in layer_names:
        test_accs = []
        
        for ber in ber_vector:
            noisy_models = add_digital_noise_to_model_brevitas(model, [layer], ber, num_perturbations)
            
            accuracies = []
            
            for noisy_model in noisy_models:
                
                noisy_model.to(device)
                accuracies.append(test(noisy_model, test_quantized_loader, device))
                
            avg_accuracy = sum(accuracies) / len(accuracies)
            
            test_accs.append(avg_accuracy)
            
            print("BER Value: {}\tAverage Accuracy: {}".format(ber, avg_accuracy))
            
        all_test_accs.append(test_accs)
        
        plt.plot(ber_vector, test_accs,
                 label='{} Accuracy at Different Perturbation Levels'.format(layer))
        
        plt.xlabel('Standard Deviation')
        plt.ylabel('Test Accuracy')
        plt.title('Effect of Noise on Test Accuracy')
        plt.legend()
        plt.savefig("noise_plots_brevitas/ber_noise/{}.png".format(layer))
        plt.clf()
        print('Done with Plot {}'.format(layer))
        
    avg_test_accs = [sum(x) / len(x) for x in zip(*all_test_accs)]
    
    plt.plot(ber_vector, avg_test_accs, label='Average',
             linewidth=3, linestyle='--', color="black")
    
    plt.xlabel('BER Value')
    
    plt.ylabel('Test Accuracy')
    
    plt.title('Effect of BER Noise on Test Accuracy (Average)')
    
    plt.legend()
    plt.savefig("noise_plots_brevitas/ber_noise/average.png")
    plt.clf()




## Digital Noise with BER
layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'fc1', 'fc2']
ber_vals = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]
perturbations = 10

ber_noise_plot_brevitas(perturbations, layer_names, ber_vals, model, device, test_quantized_loader)


## Gaussian Noise
sigma_vector = np.linspace(0, 0.2, 11)

gaussian_noise_plots_brevitas(perturbations, layer_names, sigma_vector, model, device, test_quantized_loader)

