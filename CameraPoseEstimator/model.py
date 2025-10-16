import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision

class CameraPoseEstimator(nn.Module):
    def __init__(self):
        super(CameraPoseEstimator, self).__init__()
        self.encoder = torchvision.models.resnet18(weights=True)
        self.encoder.fc = nn.Identity()  
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x