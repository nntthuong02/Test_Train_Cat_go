import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os

from game_logic import GoGame, MCTS
from model import GoNet

def generate_self_play_games(num_games=100, simulations=50):
    data = []
    mcts = MCTS(simulations=simulations)

    print(f"Generating {num_games} self-play games...")
    for _ in tqdm(range(num_games)):
        game = GoGame()
        game_data = []

        while not game.is_game_over():
            state = game.get_state()
            policy = mcts.search(game)
            
            # Simple move selection based on policy
            move = np.random.choice(len(policy), p=policy)
            game_data.append((state, policy, game.current_player))
            game.play(move)

        winner = game.get_winner()

        for state, policy, player in game_data:
            # Value is 1 if player won, -1 if player lost, 0 for draw
            value = 1 if player == winner else -1 if winner != 0 else 0
            data.append((state, policy, value))

    return data

def train(num_games=1000, simulations=400, epochs=10, batch_size=64, lr=0.001):
    # 1. Generate data
    raw_data = generate_self_play_games(num_games=num_games, simulations=simulations)
    
    states = torch.tensor(np.array([d[0] for d in raw_data]))
    policies = torch.tensor(np.array([d[1] for d in raw_data]))
    values = torch.tensor(np.array([d[2] for d in raw_data])).float()

    dataset = TensorDataset(states, policies, values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = GoNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3. Training loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for s, p, v in loader:
            s = s.to(device)
            p = p.to(device)
            v = v.to(device).unsqueeze(1)

            pred_p, pred_v = model(s)

            # Policy loss (cross entropy)
            loss_p = F.cross_entropy(pred_p, torch.argmax(p, dim=1))
            # Value loss (mean squared error)
            loss_v = F.mse_loss(pred_v, v)

            loss = loss_p + loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # 4. Save model
    model_path = "go5x5_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Parameters for a "Serious" training run
    # num_games: 500 (More samples)
    # simulations: 200 (Better move quality)
    # epochs: 20 (Stronger learning)
    train(num_games=500, simulations=200, epochs=20, batch_size=128, lr=0.0005)
