import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*5*5, 128)

        self.policy_head = nn.Linear(128, 25)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GoNet().to(device)
model.load_state_dict(torch.load("go5x5_model.pth", map_location=device))
model.eval()

print("Model loaded successfully!")

dummy_state = np.zeros((3,5,5), dtype=np.float32)
dummy_state = torch.tensor(dummy_state).unsqueeze(0).to(device)

with torch.no_grad():
    policy, value = model(dummy_state)

print("Policy:", torch.softmax(policy, dim=1))
print("Value:", value)