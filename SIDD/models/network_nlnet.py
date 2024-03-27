import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init

class NoiseLevelNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.net = nn.Sequential(
                nn.Conv2d(in_channels, 64, 4, 2, 1), 
                nn.ReLU(True),
                nn.Conv2d(64, 128, 4, 2, 1), 
                nn.ReLU(True),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.ReLU(True),
                nn.Conv2d(256, 512, 4, 2, 1), 
                nn.AdaptiveAvgPool2d(1), 
                )
        
        self.dense = nn.Sequential(
                nn.Linear(512, 256), 
                nn.ReLU(True), 
                nn.Linear(256, 128), 
                nn.ReLU(True), 
                nn.Linear(128, 1), 
                )
        
    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.dense(x).unsqueeze(2).unsqueeze(3)