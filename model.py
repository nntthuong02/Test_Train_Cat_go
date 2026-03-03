import torch
import torch.nn as nn
import torch.nn.functional as F

class GoNet(nn.Module):
    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * size * size, 128)

        self.policy_head = nn.Linear(128, size * size)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (N, 3, size, size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value
