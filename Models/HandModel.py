"""
This is the NN Model for Hand Spread Detection.
I put this into its own file to avoid issues with multiple definitions of essentially the same thing.
Used by HandModelTraining.py and HandDetection.py
"""
import torch
import torch.nn as nn

class GestureModel(nn.Module):
    def __init__(self, in_feat):
        super(GestureModel, self).__init__()
        self.fc1 = nn.Linear(in_feat * 3, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

def landmarkMask(_):
    """
    return which landmarks are to be used
    :return: list landmarks to be used
    """
    return [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]