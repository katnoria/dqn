import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    """Policy model"""

    def __init__(self, state_size, action_size, seed):
        """Initialise the linear model

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(LinearModel, self).__init__()
        self.seed = torch.manual_seed(seed)

        # define network
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)


    def forward(self, state):
        """Forward pass of network

        Params
        ======
            state (array_like): environment state
        """
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        return self.fc3(out)