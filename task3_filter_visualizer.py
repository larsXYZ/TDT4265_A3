import matplotlib.pyplot as plt
import torch
import pickle
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data.sampler import SequentialSampler
from dataloaders import load_cifar10
import task3_net
import numpy as np
import torch
from utils import to_cuda

#Visualizes filters for task 3e and 3f

#Plots a pytorch tensor as an image. Since the tensor is [ch, width, height] and not [width, height, ch] we transpose
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()
    plt.clf()

#Runs an image through the first feature extracting layer of the trained model
def first_layer_activations(model, image):
    image = nn.functional.interpolate(image , scale_factor =8)
    image = model.model.conv1(image)
    return image

#Runs an image through to the last convolutional layer.
def last_layer_activations(model, image):
    image = nn.functional.interpolate(image , scale_factor =8)
    x = image
    i = 0
    for child in model.model.children():
        print(child, i)
        x = child(x)

        i = i + 1
        if (i > 7): return x

#Loading trained model
model = task3_net.task3_model()
model = torch.load("task3_results_plots_model/trained_model")
model.eval()

#Variables
batch_size = 1
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

#We define two dataloaders. One with normalization and one without.
#This is in order to show the original image while using the normalized one in the network.
transform_normalized = [transforms.ToTensor(), transforms.Normalize(mean,std)]
transform_normalized = transforms.Compose(transform_normalized)
data_train_normalized = datasets.CIFAR10('data/cifar10',
                                train=True,
                                download=True,
                                transform=transform_normalized)

transform = [transforms.ToTensor()]
transform = transforms.Compose(transform)
data_train = datasets.CIFAR10('data/cifar10',
                                train=True,
                                download=True,
                                transform=transform)



indices = list(range(len(data_train_normalized)))
train_sampler = SequentialSampler(indices)

dataloader_train_normalized = torch.utils.data.DataLoader(data_train_normalized,
                                                sampler=train_sampler,
                                                batch_size=batch_size,
                                                num_workers=2)

dataloader_train = torch.utils.data.DataLoader(data_train,
                                                sampler=train_sampler,
                                                batch_size=batch_size,
                                                num_workers=2)


#Showing the original image
for batch_it, (X_batch, Y_batch) in enumerate(dataloader_train):
    
    image = X_batch[0,:,:,:]
    show(image)
    break

#Showing the activations after the first layer
for i in range(0,10):
    for batch_it, (X_batch, Y_batch) in enumerate(dataloader_train_normalized):
        
        #image_features = first_layer_activations(model, to_cuda(X_batch)).cpu() , used for task 3e
        image_features = last_layer_activations(model, to_cuda(X_batch)).cpu() # used for task 3f

        print(image_features.size())
        plt.imshow(image_features[0,i,:,:].detach())
        plt.show()

        break
    