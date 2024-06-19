"""
Model used for arm direction detection.
First draft, based on response by chatgpt.
To be optimised.
"""
import torch
import torch.nn as nn

class PoseGestureModel(nn.Module):
    def __init__(self, in_feat):
        super(PoseGestureModel, self).__init__()
        self.fc1 = nn.Linear(in_feat * 3, 32)  # in_feat (num_landmarks) * 3 dimensions
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

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
        return [0, 7, 8, 11, 13, 15, 23, 24]
    else:
        raise Exception("ARM must be 'left' or 'right'!")
