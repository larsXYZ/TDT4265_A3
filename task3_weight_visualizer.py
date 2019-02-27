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

#Loading trained model
model = task3_net.task3_model()
model = torch.load("task3_results_plots_model/trained_model")
model.eval()


for param in model.model.conv1.parameters():
  for i in range(64):
        
        weights = param[i,:,:,:].cpu()
        
        min_val = torch.min(weights)
        weights = weights - min_val*torch.ones(weights.size())
        max_val = torch.max(weights)
        weights = weights/max_val
        
        print(weights.size())
        print(weights)
        show(weights)

