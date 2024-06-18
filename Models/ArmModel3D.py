"""
Model used for 3D arm direction detection.
First draft, based on response by chatgpt.
To be optimised.
"""
import torch
import torch.nn as nn

class PoseGestureModel(nn.Module):
    def __init__(self):
        super(PoseGestureModel, self).__init__()
        self.fc1 = nn.Linear(24, 32) # 8 input landmarks * 3 dimensions = 24 in features
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

def landmarkMask(arm):
    """
    return which landmarks are to be used for each arm
    :param arm: str 'left' or 'right'
    :return: list landmarks to be used
    """

    if arm.lower() == 'right':
        return [0, 7, 8, 12, 14, 16, 23, 24]
    elif arm.lower() == 'left':
        return [0, 7, 8, 11, 13, 15, 23, 23]
    else:
        raise Exception("ARM must be 'left' or 'right'!")
