import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy


class task1_model(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()


        # First CNN layer
        self.first_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64)

        )

        # Second CNN layer
        self.second_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        # Third CNN layer
        self.third_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        #First dense layer
        self.fourth_layer = nn.Sequential(
            nn.Linear(256*16, 64),
            nn.ReLU()
        )

        #Second dense layer
        self.fifth_layer = nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        #CNN
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = self.third_layer(x)

        x = x.view(-1, 256*4*4)

        #Dense
        x = self.fourth_layer(x)
        x = self.fifth_layer(x)


        return x
