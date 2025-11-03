import torch 
import torch.nn.functional as F 
from shift_invariant_unet.pbpunet.pbpunet_model import PBPUNet 
import torchvision.transforms.functional as TF
import model_utils

class AgeGAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = PBPUNet(6, 3)
        

    def forward(self, input_images, landmark):
        # x = model_utils.append_age_channel(input_images, target_ages)
        x = torch.cat([input_images, landmark], dim=1)
        output = self.unet(x.float())
        # output = F.sigmoid(output)
        output = F.tanh(output)
        
        return output

