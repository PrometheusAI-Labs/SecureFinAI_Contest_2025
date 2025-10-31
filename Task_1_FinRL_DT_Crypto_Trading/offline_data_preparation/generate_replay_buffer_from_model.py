#!/usr/bin/env python3
"""
Generate replay buffer from trained model by running inference

This script loads a trained RL agent model and collects trajectories
by running it in the environment, then saves the data in replay buffer format.
"""

import os
import sys
import torch
import numpy as np
from erl_config import Config, build_env
from erl_replay_buffer import ReplayBuffer
from erl_agent import AgentD3QN
from trade_simulator import TradeSimulator


def load_trained_agent(model_dir: str, gpu_id: int = 0):
    """
    Load trained agent from model directory
    
    Args:
        model_dir: Directory containing trained model files
        gpu_id: GPU ID to use
        
    Returns:
        Trained agent instance
    """
    print(f"Loading trained agent from: {model_dir}")
    
    # Extract agent class and seed from directory name
    # Format: TradeSimulator-v0_AgentName_seed
    dir_name = os.path.basename(model_dir.rstrip('/'))
    parts = dir_name.split('_')
    
    if len(parts) < 3:
        raise ValueError(f"Unexpected directory name format: {dir_name}")
    
    agent_name = parts[1]  # e.g., "D3QN", "DoubleDQN", "TwinD3QN"
    
    # Map agent names to classes
    from erl_agent import AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
    agent_class_map = {
        'D3QN': AgentD3QN,
        'DoubleDQN': AgentDoubleDQN,
        'TwinD3QN': AgentTwinD3QN,
    }
    
    agent_class = agent_class_map.get(agent_name, AgentD3QN)
    print(f"Detected agent class: {agent_class.__name__}")
    
    # Model parameters (should match training config)
    num_sims = 2**11  # 2048
    num_ignore_step = 60
    step_gap = 2
    max_step = (4800 - num_ignore_step) // step_gap
    
    env_args = {
        "env_name": "TradeSimulator-v0",
        "num_envs": num_sims,
        "max_step": max_step,
        "state_dim": 2 + 8 + 2,
        "action_dim": 3,
        "if_discrete": True,
        "max_position": 1,
        "slippage": 7e-7,
        "num_sims": num_sims,
        "step_gap": step_gap,
    }
    
    args = Config(agent_class=agent_class, env_class=TradeSimulator, env_args=env_args)
    args.gpu_id = gpu_id
    args.net_dims = (128, 128, 128)
    
    # Create agent
    agent = agent_class(
        args.net_dims,
        args.state_dim,
        args.action_dim,
        gpu_id=gpu_id,
        args=args,
    )
    
    # Load trained weights
    agent.save_or_load_agent(model_dir, if_save=False)
    
    # Load the best/latest actor checkpoint if available
    actor_files = sorted(
        [f for f in os.listdir(model_dir) if f.startswith("actor_") and f.endswith(".pth")]
    )
    if actor_files:
        latest_path = os.path.join(model_dir, actor_files[-1])
        print(f"Loading latest actor checkpoint: {actor_files[-1]}")
        loaded = torch.load(latest_path, map_location=agent.device, weights_only=False)
        state_dict = loaded.state_dict() if hasattr(loaded, "state_dict") else loaded
        agent.act.load_state_dict(state_dict)
        agent.act_target.load_state_dict(state_dict)
    else:
        print("Using act.pth (final saved model)")
    
    return agent, args


def collect_trajectories(agent, env, num_collection_rounds: int = 100, horizon_len: int = None, use_cpu_buffer: bool = True):
    """
    Collect trajectories from trained agent
    
    Args:
        agent: Trained agent
        env: Environment
        num_collection_rounds: Number of collection rounds (each round collects horizon_len steps)
        horizon_len: Steps per collection round (defaults to max_step * 2, reduced for memory)
        use_cpu_buffer: Use CPU for buffer storage to avoid OOM
        
    Returns:
        ReplayBuffer with collected data
    """
    if horizon_len is None:
        horizon_len = env.max_step * 2  # Reduced from *4 to *2 for memory efficiency
    
    # Calculate reasonable buffer size (limit to avoid OOM)
    # Use smaller buffer size, we'll collect in rounds
    max_buffer_size = min(horizon_len * num_collection_rounds, 50000)  # Limit to ~50k steps
    
    print(f"Buffer settings: max_size={max_buffer_size}, num_seqs={agent.num_envs}, state_dim={agent.state_dim}")
    print(f"Estimated buffer memory: {max_buffer_size * agent.num_envs * agent.state_dim * 4 / 1e9:.2f} GB")
    
    # Initialize replay buffer on CPU to avoid OOM
    buffer = ReplayBuffer(
        gpu_id=-1 if use_cpu_buffer else (agent.device.index if agent.device.type == 'cuda' else -1),
        num_seqs=agent.num_envs,
        max_size=max_buffer_size,
        state_dim=agent.state_dim,
        action_dim=1,  # Discrete actions
    )
    
    # Initialize agent state
    state = env.reset()
    if isinstance(state, np.ndarray) and state.ndim == 1:
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    else:
        state = state.to(agent.device) if isinstance(state, torch.Tensor) else torch.tensor(state, device=agent.device)
    agent.last_state = state.detach()
    
    print(f"Collecting trajectories: {num_collection_rounds} rounds × {horizon_len} steps = {num_collection_rounds * horizon_len} total steps")
    
    for round_idx in range(num_collection_rounds):
        if (round_idx + 1) % 10 == 0:
            print(f"  Collection round {round_idx + 1}/{num_collection_rounds}... (buffer: {buffer.cur_size}/{buffer.max_size})")
        
        # Collect one round of data
        buffer_items = agent.explore_env(env, horizon_len, if_random=False)
        
        # Move to CPU if buffer is on CPU
        if use_cpu_buffer:
            buffer_items = tuple(
                item.cpu() if isinstance(item, torch.Tensor) else item
                for item in buffer_items
            )
        
        # Update buffer (will handle wrapping if full)
        buffer.update(buffer_items)
        
        # If buffer is full, we'll wrap around (FIFO)
        if buffer.if_full:
            print(f"  Buffer full at {buffer.cur_size} steps, wrapping around (FIFO)")
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and (round_idx + 1) % 20 == 0:
            torch.cuda.empty_cache()
    
    print(f"Collection complete! Buffer size: {buffer.cur_size}/{buffer.max_size}")
    return buffer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate replay buffer from trained model")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing trained model (e.g., ./TradeSimulator-v0_D3QN_0)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save replay buffer (defaults to model_dir)")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID to use")
    parser.add_argument("--num_rounds", type=int, default=100,
                       help="Number of collection rounds (default: 100)")
    args = parser.parse_args()
    
    model_dir = args.model_dir
    output_dir = args.output_dir if args.output_dir else model_dir
    gpu_id = args.gpu_id
    
    # Verify model directory exists
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Check for required files
    if not os.path.isfile(os.path.join(model_dir, "act.pth")):
        raise FileNotFoundError(f"Model file not found: {os.path.join(model_dir, 'act.pth')}")
    
    print(f"🚀 Generating replay buffer from trained model")
    print(f"   Model directory: {model_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   GPU ID: {gpu_id}")
    print()
    
    # Load trained agent
    agent, config_args = load_trained_agent(model_dir, gpu_id=gpu_id)
    
    # Create environment
    env = build_env(config_args.env_class, config_args.env_args, gpu_id=gpu_id)
    
    # Collect trajectories (use CPU buffer to avoid OOM)
    buffer = collect_trajectories(agent, env, num_collection_rounds=args.num_rounds, use_cpu_buffer=True)
    
    # Save replay buffer
    print(f"\n💾 Saving replay buffer to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear GPU memory before saving
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    try:
        buffer.save_or_load_history(output_dir, if_save=True)
        print("✅ Replay buffer saved successfully!")
    except torch.cuda.OutOfMemoryError as e:
        print(f"⚠️  Warning: OOM during save. Trying CPU offload...")
        # Try saving on CPU
        buffer.device = torch.device("cpu")
        for attr in ['states', 'actions', 'rewards', 'undones']:
            if hasattr(buffer, attr):
                setattr(buffer, attr, getattr(buffer, attr).cpu())
        buffer.save_or_load_history(output_dir, if_save=True)
        print("✅ Replay buffer saved on CPU!")
    
    env.close() if hasattr(env, "close") else None
    print(f"\n✅ Done! Replay buffer files saved in: {output_dir}")


if __name__ == "__main__":
    main()

