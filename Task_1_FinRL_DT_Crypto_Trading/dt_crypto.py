#!/usr/bin/env python3
"""
Train a Decision Transformer for Crypto Trading

This script trains a Decision Transformer model for crypto trading.
It uses return-to-go conditioning and causal action shifting.

Usage:
    python dt_crypto.py [--epochs NUM] [--lr FLOAT] [--context_length NUM] [--model_path PATH] [--plots_dir DIR]

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DecisionTransformerConfig, DecisionTransformerModel
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ast
import os
import time

# Check MPS availability
if torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders) is available on this Mac")
else:
    print("MPS is not available, will use CPU")

def parse_array(s):
    """
    Parses a string representation of a numpy array or list.
    Handles the format used in crypto dataset.
    """
    try:
        parsed = ast.literal_eval(s)
        return np.array(parsed)
    except Exception as e:
        print(f"FATAL ERROR: Could not parse string: '{s}'")
        raise e

class CryptoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, context_len: int):
        self.df = df.copy()
        self.context_len = context_len
        self.df['episode_start'] = self.df['episode_start'].astype(bool)
        self.ep_starts = self.df.index[self.df['episode_start']].tolist()
        self.ep_ends = self.ep_starts[1:] + [len(self.df)]

        # precompute (episode_base, offset) so windows never cross episodes
        self.starts = []
        for s, e in zip(self.ep_starts, self.ep_ends):
            L = e - s
            for offset in range(L):
                self.starts.append((s, offset))

        first_state = parse_array(self.df['state'].iloc[0])
        self.state_dim = len(first_state)
        self.action_dim = 3

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        ep_base, offset = self.starts[idx]
        # window = [offset-K+1 ... offset] inside the episode; left-pad if needed
        left = max(0, offset - (self.context_len - 1))
        right = offset + 1
        seg = self.df.iloc[ep_base + left : ep_base + right]
        T = len(seg)
        pad = self.context_len - T

        states_np = np.vstack([parse_array(s) for s in seg['state'].values])  # (T, S)
        states = torch.from_numpy(states_np).float()

        a_scalar = seg['action'].values.astype(int)
        
        # Normalize actions to 0-2 range if needed
        unique_actions = np.unique(a_scalar)
        if len(unique_actions) == 2:
            # Map 2 actions to hold(1) and buy(2) - most common trading actions
            a_scalar = np.where(a_scalar == min(unique_actions), 1, 2)  # 0->1(hold), 1->2(buy)
        elif len(unique_actions) == 3 and max(unique_actions) > 2:
            # Map actions to 0-2 range if we have 3 actions
            a_scalar = a_scalar - min(unique_actions)
        elif max(unique_actions) > 2:
            # General case: map to 0-2 range
            a_scalar = np.clip(a_scalar, 0, 2)
        
        actions = torch.zeros(T, self.action_dim, dtype=torch.float32)
        actions[torch.arange(T), a_scalar] = 1.0

        rewards = torch.from_numpy(seg['reward'].values).float().view(-1, 1)
        rtg = torch.from_numpy(seg['return_to_go'].values).float().view(-1, 1)
        timesteps = torch.arange(left, left + T, dtype=torch.long)  # episode-local

        if pad > 0:
            states = torch.cat([torch.zeros(pad, states.size(1)), states], dim=0)
            actions = torch.cat([torch.zeros(pad, self.action_dim), actions], dim=0)
            rewards = torch.cat([torch.zeros(pad, 1), rewards], dim=0)
            rtg = torch.cat([torch.zeros(pad, 1), rtg], dim=0)
            timesteps = torch.cat([torch.zeros(pad, dtype=torch.long), timesteps], dim=0)

        attn = torch.zeros(self.context_len, dtype=torch.long)
        attn[-T:] = 1

        return {
            "states": states, 
            "actions": actions, 
            "returns_to_go": rtg, 
            "timesteps": timesteps, 
            "attention_mask": attn
        }


def train(epochs, lr, context_length, model_path, plots_dir):
    print("--- Starting Training ---")
    start_time = time.time()
    df = pd.read_csv("crypto_decision_transformer_ready_dataset.csv")

    df['episode_start'] = df['episode_start'].astype(bool)
    episode_starts = df.index[df['episode_start']].tolist()
    episode_indices = list(range(len(episode_starts)))
    
    # Create proper train/validation/test split: 60%/20%/20%
    train_ep_indices, temp_ep_indices = train_test_split(episode_indices, test_size=0.4, random_state=42)
    val_ep_indices, test_ep_indices = train_test_split(temp_ep_indices, test_size=0.5, random_state=42)
    
    print(f"Dataset split: {len(train_ep_indices)} train episodes, {len(val_ep_indices)} val episodes, {len(test_ep_indices)} test episodes")
    print(f"Split ratios: {len(train_ep_indices)/len(episode_indices):.1%} train, {len(val_ep_indices)/len(episode_indices):.1%} val, {len(test_ep_indices)/len(episode_indices):.1%} test")

    train_df_indices = []
    for ep_idx in train_ep_indices:
        start = episode_starts[ep_idx]
        end = episode_starts[ep_idx+1] if ep_idx + 1 < len(episode_starts) else len(df)
        train_df_indices.extend(range(start, end))

    train_df = df.iloc[train_df_indices].reset_index(drop=True)

    val_df_indices = []
    for ep_idx in val_ep_indices:
        start = episode_starts[ep_idx]
        end = episode_starts[ep_idx+1] if ep_idx + 1 < len(episode_starts) else len(df)
        val_df_indices.extend(range(start, end))
    val_df = df.iloc[val_df_indices].reset_index(drop=True)

    test_df_indices = []
    for ep_idx in test_ep_indices:
        start = episode_starts[ep_idx]
        end = episode_starts[ep_idx+1] if ep_idx + 1 < len(episode_starts) else len(df)
        test_df_indices.extend(range(start, end))
    test_df = df.iloc[test_df_indices].reset_index(drop=True)
    
    # Save test set for final evaluation
    test_df.to_csv("crypto_test_dataset.csv", index=False)
    print(f"💾 Test dataset saved to: crypto_test_dataset.csv")
    print(f"   Test episodes: {len(test_ep_indices)}")
    print(f"   Test samples: {len(test_df)}")

    # Set device with MPS support for Mac
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU for training")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        try:
            # Test MPS with a simple operation
            test_tensor = torch.randn(1, 1).to("mps")
            device = torch.device("mps")
            print("🚀 Using MPS (Metal Performance Shaders) on Mac")
            print("   This should provide significant speedup on M1/M2 Macs")
        except Exception as e:
            print(f"❌ MPS test failed: {e}")
            print("⚠️  Falling back to CPU - training will be slower")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU - training will be slower")
        print("   Consider using a Mac with M1/M2 chip for MPS acceleration")

    # Address Class Imbalance 
    class_counts = train_df['action'].value_counts().sort_index()
    total_samples = class_counts.sum()
    num_classes = len(class_counts)
    
    # Ensure we have weights for all possible classes (0, 1, 2)
    class_weights = np.zeros(3)  # Always use 3 classes for action space
    for i in range(3):
        if i in class_counts.index:
            class_weights[i] = total_samples / (num_classes * class_counts[i])
        else:
            class_weights[i] = 1.0  # Default weight for missing classes
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Using Class Weights to combat imbalance: {class_weights.cpu().numpy()}")
    print(f"Available action classes: {sorted(class_counts.index.tolist())}")
    print(f"Action dimension: {num_classes} (mapped to 3 classes)")

    train_dataset = CryptoDataset(train_df, context_len=context_length)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Reduced batch size

    val_dataset = CryptoDataset(val_df, context_len=context_length)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Reduced batch size

    state_dim = train_dataset.state_dim
    act_dim = train_dataset.action_dim

    print(f"State dimension: {state_dim}, Action dimension: {act_dim}")

    config = DecisionTransformerConfig(
        state_dim=state_dim, 
        act_dim=act_dim, 
        hidden_size=64,
        n_layer=3,
        n_head=1, 
        n_inner=4*64,
        resid_pdrop=0.3,
        attn_pdrop=0.3,
        action_tanh=False
    )
    model = DecisionTransformerModel(config)
    model = model.to(device)

    from torch.optim.lr_scheduler import StepLR
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.6)  # Further increased weight decay from 0.5
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    CLIP_GRAD_NORM = 2.0

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            states, actions, returns_to_go, timesteps, attention_mask = (
                batch["states"].to(device), batch["actions"].to(device),
                batch["returns_to_go"].to(device), batch["timesteps"].to(device),
                batch["attention_mask"].to(device)
            )
            
            optimizer.zero_grad()
            
            # shift actions right: input previous actions, target current actions
            actions_in = torch.zeros_like(actions)
            actions_in[:, 1:, :] = actions[:, :-1, :]  # feed a_{t-1}

            out = model(
                states=states,
                actions=actions_in,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask
            )
            logits = out.action_preds  # (B, T, 3)

            # -- targets as class ids (0/1/2), mask out padding --
            targets = actions.argmax(dim=-1)      # (B, T)
            mask = attention_mask.bool()          # (B, T)

            loss = F.cross_entropy(
                logits[mask],     # (N, 3)
                targets[mask],    # (N,)
                weight=class_weights, # Apply class weights
                label_smoothing=0.1  # Add label smoothing to reduce overfitting
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()    
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                states, actions, returns_to_go, timesteps, attention_mask = (
                    batch["states"].to(device), batch["actions"].to(device),
                    batch["returns_to_go"].to(device), batch["timesteps"].to(device),
                    batch["attention_mask"].to(device)
                )
                
                # shift actions right: input previous actions, target current actions
                actions_in = torch.zeros_like(actions)
                actions_in[:, 1:, :] = actions[:, :-1, :]  # feed a_{t-1}

                out = model(
                    states=states,
                    actions=actions_in,
                    returns_to_go=returns_to_go,
                    timesteps=timesteps,
                    attention_mask=attention_mask
                )
                logits = out.action_preds  # (B, T, 3)

                # -- targets as class ids (0/1/2), mask out padding --
                targets = actions.argmax(dim=-1)      # (B, T)
                mask = attention_mask.bool()          # (B, T)

                val_loss = F.cross_entropy(
                    logits[mask],     # (N, 3)
                    targets[mask],    # (N,)
                    weight=class_weights, # Apply class weights
                    label_smoothing=0.1  # Add label smoothing to reduce overfitting
                )
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}/{epochs}, Avg Train Loss: {avg_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    end_time = time.time()
    training_time = end_time - start_time
    
    print("--- Training Finished ---")
    print(f"⏱️  Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"📊 Final validation loss: {best_val_loss:.4f}")
    print(f"🎯 Model saved to: {model_path}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training & Validation Loss - Crypto Decision Transformer')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'training_loss_plot_crypto.png'))
    print(f"📈 Loss plot saved to {os.path.join(plots_dir, 'training_loss_plot_crypto.png')}")

def preprocess_raw_btc_data(raw_df, window_size=10):
    """
    Convert raw BTC data (156 features) to 12-dimensional state representation.
    Based on analysis of the training data patterns.
    """
    import numpy as np
    
    # Extract key features to match the 12-dimensional state
    features = []
    
    # Feature engineering to create 12-dimensional state
    # Based on patterns observed in training data
    
    for i in range(len(raw_df)):
        state = []
        
        # Dimension 0: Action proxy (based on buy/sell activity)
        buys = raw_df.iloc[i]['buys']
        sells = raw_df.iloc[i]['sells']
        if buys > sells:
            action_proxy = 1.0  # Buy signal
        elif sells > buys:
            action_proxy = -1.0  # Sell signal
        else:
            action_proxy = 0.0  # Hold signal
        state.append(action_proxy)
        
        # Dimension 1: Spread normalized
        spread = raw_df.iloc[i]['spread']
        midpoint = raw_df.iloc[i]['midpoint']
        spread_norm = spread / midpoint if midpoint > 0 else 0.0
        state.append(spread_norm)
        
        # Dimensions 2-9: Order book features (normalized)
        # Use bid/ask distances and notionals as indicators
        bid_features = []
        ask_features = []
        
        # Aggregate bid book features
        for j in range(5):  # Use first 5 levels
            bid_dist = raw_df.iloc[i].get(f'bids_distance_{j}', 0.0)
            bid_notional = raw_df.iloc[i].get(f'bids_notional_{j}', 0.0)
            bid_features.extend([bid_dist, bid_notional])
            
            ask_dist = raw_df.iloc[i].get(f'asks_distance_{j}', 0.0)
            ask_notional = raw_df.iloc[i].get(f'asks_notional_{j}', 0.0)
            ask_features.extend([ask_dist, ask_notional])
        
        # Compute summary statistics for dimensions 2-9
        bid_mean = np.mean(bid_features) if bid_features else 0.0
        ask_mean = np.mean(ask_features) if ask_features else 0.0
        bid_std = np.std(bid_features) if len(bid_features) > 1 else 0.0
        ask_std = np.std(ask_features) if len(ask_features) > 1 else 0.0
        
        # Market activity indicators
        bid_volume = sum([raw_df.iloc[i].get(f'bids_notional_{j}', 0.0) for j in range(5)])
        ask_volume = sum([raw_df.iloc[i].get(f'asks_notional_{j}', 0.0) for j in range(5)])
        volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)
        
        # Price momentum 
        if i > 0:
            price_change = (midpoint - raw_df.iloc[i-1]['midpoint']) / raw_df.iloc[i-1]['midpoint']
        else:
            price_change = 0.0
        
        # Add the 8 financial features (dimensions 2-9)
        state.extend([
            bid_mean, ask_mean, bid_std, ask_std,
            volume_imbalance, price_change, 
            np.log(bid_volume + 1), np.log(ask_volume + 1)
        ])
        
        # Dimensions 10-11: Sentiment and risk scores
        sentiment_score = raw_df.iloc[i]['sentiment_score']
        risk_score = raw_df.iloc[i]['risk_score']
        state.extend([sentiment_score, risk_score])
        
        features.append(state)
    
    return np.array(features)

  
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a Decision Transformer for Crypto Trading')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--context_length', type=int, default=20, help='Context length for the transformer')
    parser.add_argument('--model_path', type=str, default='./trained_models/decision_transformer.pth', help='Path to save the trained model')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Directory to save training plots')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Train the model
    print(f"🚀 Training Decision Transformer for Crypto Trading")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Context length: {args.context_length}")
    print(f"   Model will be saved to: {args.model_path}")
    print(f"   Training plots will be saved to: {args.plots_dir}")
    print()
    
    train(
        epochs=args.epochs,
        lr=args.lr,
        context_length=args.context_length,
        model_path=args.model_path,
        plots_dir=args.plots_dir
    )
