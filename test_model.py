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

# Test Case: Empty board, Black to move (player = 1.0)
# We want to see if the AI likes the center (index 12)
dummy_state = np.zeros((3, 5, 5), dtype=np.float32)
dummy_state[2, :, :] = 1.0  # Current player = 1.0 (Black)

# Convert to tensor and add batch dimension
state_input = torch.tensor(dummy_state).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    policy_logits, value = model(state_input)

# Convert logits to probabilities
policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
best_move = np.argmax(policy_probs)
best_prob = policy_probs[best_move]

# Board mapping (optional visualization)
print("\n--- Inference Results ---")
print(f"Predicted Value: {value.item():.4f} (positive = Black advantage)")
print(f"Best Move Index: {best_move} (Row {best_move // 5}, Col {best_move % 5})")
print(f"Best Move Confidence: {best_prob * 100:.2f}%")

if best_move == 12:
    print("✅ Success: AI prioritized the center (Tengen)!")
else:
    print(f"ℹ️ Note: AI prioritized index {best_move} instead of center 12.")
    print("This is normal if training is still in early stages.")

# Visualizing policy on 5x5 grid
print("\nPolicy Distribution (5x5):")
print(policy_probs.reshape(5, 5).round(3))