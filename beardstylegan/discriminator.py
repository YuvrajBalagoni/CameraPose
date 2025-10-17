import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters

import model_utils

class BlurPool(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
    
    def forward(self, x):
        return kornia.filters.blur_pool2d(x, self.stride)


class Discriminator(nn.Module):

    def __init__(self, in_channels, out_channels=1):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            self.discriminator_block(in_channels, 64, normalize=False),
            self.discriminator_block(64, 128),
            self.discriminator_block(128, 256),
            self.discriminator_block(256, 512),
            self.discriminator_block(512, 1, normalize=False),
            nn.Sigmoid(),
        )

    # def forward(self, images1, images2, ages):
    #    x = model_utils.append_image_channel(images1, images2)
    #    x = model_utils.append_age_channel(x, ages) 
    #    return self.layers(x.float())
    
    def forward(self, images):
    #    x = model_utils.append_age_channel(images, ages) 
        x = images
        return self.layers(x.float())

    def discriminator_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, normalize=True):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1),
            BlurPool(stride),
            *([nn.BatchNorm2d(num_features=out_channels)] if normalize else []),
            nn.LeakyReLU(0.2, inplace=True) )




