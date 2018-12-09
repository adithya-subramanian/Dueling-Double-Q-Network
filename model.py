import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DDDQNetwork(nn.Module):
    """Dueling Double Deep Q Network Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DDDQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(self.state_size,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,64)
        self.fc6 = nn.Linear(64,32)
        self.fc7_1 = nn.Linear(32,16)
        self.fc8_1 = nn.Linear(16,1)
        self.fc7_2 = nn.Linear(32,16)
        self.fc8_2 = nn.Linear(16,self.action_size)
        self.act = nn.ELU()
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(32)
        self.fc1_skip = nn.Linear(1024,256)
        self.fc2_skip = nn.Linear(256,64)

    def forward(self, state):
        """Build a network that maps state -> action values.
        
        The network is built similar to that of the resent except exponential linear unit is used as activations instead of Recitified linear unit.
        """
        x = self.fc1(state)
        x = self.act(x)
        x = self.bn1(x)
        x1 = self.fc2(x)
        x1 = self.act(x1)
        x1 = self.bn2(x1)
        x1 = self.fc3(x1)
        x1 = self.act(x1)
        x1 = self.bn3(x1)
        x = torch.add(x1,self.act(self.fc1_skip(x)))
        x1 = self.fc4(x)
        x1 = self.act(x1)
        x1 = self.bn4(x1)
        x1 = self.fc5(x1)
        x1 = self.act(x1)
        x1 = self.bn5(x1)
        x = torch.add(x1,self.act(self.fc2_skip(x)))
        x = self.fc6(x)
        x = self.act(x)
        x = self.bn6(x)
        return torch.add(self.fc8_1(self.act(self.fc7_1(x))),self.fc8_2(self.act(self.fc7_2(x))) - torch.mean(self.fc8_2(self.act(self.fc7_2(x)))))