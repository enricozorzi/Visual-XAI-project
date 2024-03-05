import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from kymatio.torch import Scattering2D

'''
ScatNet 2D
'''

class ScatNet2D(nn.Module):
    def __init__(self, input_channels: int, scattering: Scattering2D):
        super(ScatNet2D, self).__init__()
        self.k = input_channels
        self.scattering = scattering
        
        #k=217 con J=3   nn.Linear(self.k*16*16*3, 1024),
        #k=81 con J=2  nn.Linear(self.k*32*32*3, 1024),
        self.classifier = nn.Sequential(
            nn.Linear(self.k*32*32*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2), 
            )

        # Weights initialization
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight )


    def forward(self, x):
        
        x = self.scattering(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = torch.softmax(x, dim=1) # get probabilities for each class
        return x