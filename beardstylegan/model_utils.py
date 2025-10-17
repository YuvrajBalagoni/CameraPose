import torch
import os


def convert_to_tensor(var):
    if isinstance(var, float):
        var = torch.tensor([var])
    else:
        assert isinstance(var, torch.Tensor)
    return var

def append_image_channel(images1, images2):
    ndims = images1.dim()
    
    if ndims == 3:  # No batch dimension, single image
        images = torch.cat((images1, images2), 0)
    else:
        assert ndims == 4, f"Append image channel got {ndims} dimensions, expected 4."
        images = torch.cat((images1, images2), 1)
    
    return images

def append_age_channel(images, ages):
    ages = convert_to_tensor(ages)
    ndims = images.dim()
    device = images.device

    if ndims == 3:  # No batch dimension, single image
        c, h, w = images.shape
        age_channel = torch.ones([1, h, w], dtype=torch.float) * ages[:, None, None] / 10.0
        images = torch.cat((images, age_channel.to(device)), 0)
    else: 
        assert ndims == 4, f"Append age channel got {ndims} dimensions, expected 4."
        b, c, h, w = images.shape
        age_channel = torch.ones([b, 1, h, w], dtype=torch.float) * ages[:, None, None, None] / 10.0
        images = torch.cat((images, age_channel.to(device)), 1)
    return images 



