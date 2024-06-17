import torch
import torch.nn as nn

class PoseGestureModel(nn.Module):
    def __init__(self):
        super(PoseGestureModel, self).__init__()
        self.fc1 = nn.Linear(24, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x