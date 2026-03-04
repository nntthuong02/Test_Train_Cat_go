import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import pickle

from game_logic import GoGame, MCTS
from model import GoNet
import argparse

# Paths for persistence (can be overridden by environment variables for Colab/Drive)
DATA_PATH = os.getenv("GO_DATA_PATH", "training_data.pkl")
MODEL_PATH = os.getenv("GO_MODEL_PATH", "go5x5_model.pth")

def generate_self_play_games(num_games=100, simulations=50):
    data = []
    mcts = MCTS(simulations=simulations)

    for _ in tqdm(range(num_games), desc="Generating games"):
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

def save_data(data, path=DATA_PATH):
    # Append to existing data if it exists
    existing_data = []
    if os.path.exists(path):
        with open(path, "rb") as f:
            existing_data = pickle.load(f)
    
    existing_data.extend(data)
    with open(path, "wb") as f:
        pickle.dump(existing_data, f)
    return len(existing_data)

def load_data(path=DATA_PATH):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return []

def train_on_data(raw_data, epochs=10, batch_size=128, lr=0.0005):
    if not raw_data:
        print("No data to train on.")
        return

    states = torch.tensor(np.array([d[0] for d in raw_data]))
    policies = torch.tensor(np.array([d[1] for d in raw_data]))
    values = torch.tensor(np.array([d[2] for d in raw_data])).float()

    dataset = TensorDataset(states, policies, values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {len(raw_data)} samples using {device}...")
    
    model = GoNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Resuming from {MODEL_PATH}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for s, p, v in loader:
            s = s.to(device)
            p = p.to(device)
            v = v.to(device).unsqueeze(1)

            pred_p, pred_v = model(s)
            loss_p = F.cross_entropy(pred_p, torch.argmax(p, dim=1))
            loss_v = F.mse_loss(pred_v, v)
            loss = loss_p + loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def run_pipeline(target_games=100000, batch_games=500, simulations=400):
    total_data = load_data()
    current_games = len(total_data) // 20 # Approximation
    print(f"Current samples in database: {len(total_data)}")

    while current_games < target_games:
        # 1. Generate a batch of games
        batch_data = generate_self_play_games(num_games=batch_games, simulations=simulations)
        
        # 2. Save data to disk (persistence)
        total_samples = save_data(batch_data)
        current_games = total_samples // 20
        print(f"Progress: ~{current_games}/{target_games} games. Total samples: {total_samples}")

        # 3. Train on all accumulated data
        # We can also decide to train only on a subset or the whole thing
        # For simplicity, we train on everything for now
        total_data = load_data()
        train_on_data(total_data, epochs=5) # Frequent small updates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Go AI using self-play.")
    parser.add_argument("--target_games", type=int, default=1000, help="Total target games to reach.")
    parser.add_argument("--batch_games", type=int, default=100, help="Number of games to generate per batch.")
    parser.add_argument("--simulations", type=int, default=100, help="Number of MCTS simulations per move.")
    
    args = parser.parse_args()
    
    print(f"Starting training pipeline:")
    print(f"  Target Games: {args.target_games}")
    print(f"  Batch Games:  {args.batch_games}")
    print(f"  Simulations: {args.simulations}")

    run_pipeline(
        target_games=args.target_games, 
        batch_games=args.batch_games, 
        simulations=args.simulations
    )
