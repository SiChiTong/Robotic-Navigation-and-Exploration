import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO(Lab-02): Complete the network model.
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Linear(23, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, 2)

    def forward(self, s):
        output = self.layer1(s)
        output = F.relu(output)
        output = self.layer2(output)
        output = F.relu(output)
        output = self.layer3(output)
        output = F.relu(output)
        output = self.layer4(output)
        output = F.tanh(output)
        
        return output

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.layer1 = nn.Linear(23, 512)
        self.layer2 = nn.Linear(514, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, 1)
    
    def forward(self, s, a):
        output = self.layer1(s)
        output = F.relu(output)
        output = torch.cat([output,a], dim=1)
        output = self.layer2(output)
        output = F.relu(output)
        output = self.layer3(output)
        output = F.relu(output)
        output = self.layer4(output)
        
        return output