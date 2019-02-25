import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy


class task3_model(nn.Module):

    def __init__( self ):
        super () . __init__ ()
        self.model = torchvision.models.resnet18( pretrained = True )
        self.model.fc = nn.Linear( 512 *4 , 10 )

        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False

        for param in self.model.fc.parameters(): # Unfreeze the last fully - connected
            param.requires_grad = True # layer

        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers

    def forward( self , x):
        x = nn.functional.interpolate(x , scale_factor =8)
        x = self.model(x)
        return x