import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class QNetTwin(nn.Module):
    """Twin Q-Network for Double DQN"""
    
    def __init__(self, net_dims: list, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mid_dim = net_dims[0] if net_dims else 128
        
        # State normalization parameters
        self.state_avg = torch.zeros(state_dim)
        self.state_std = torch.ones(state_dim)
        
        # Value normalization parameters
        self.value_avg = torch.zeros(1)
        self.value_std = torch.ones(1)
        
        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim, self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, action_dim)
        )
        
        # Q2 network
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim, self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks
        Args:
            state: Input state tensor of shape (batch_size, state_dim)
        Returns:
            q1: Q-values from first network
            q2: Q-values from second network
        """
        q1 = self.q1_net(state)
        q2 = self.q2_net(state)
        return q1, q2
    
    def get_q_min(self, state: torch.Tensor) -> torch.Tensor:
        """Get minimum Q-value between the two networks (for conservative estimation)"""
        q1, q2 = self.forward(state)
        return torch.min(q1, q2)
    
    def get_q_max(self, state: torch.Tensor) -> torch.Tensor:
        """Get maximum Q-value between the two networks"""
        q1, q2 = self.forward(state)
        return torch.max(q1, q2)
    
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get action with highest Q-value"""
        q1, q2 = self.forward(state)
        q_values = torch.min(q1, q2)  # Conservative estimation
        return q_values.argmax(dim=1, keepdim=True)
    
    def get_q1_q2(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q1 and Q2 values separately"""
        return self.forward(state)


class QNetTwinDuel(nn.Module):
    """Dueling Twin Q-Network for D3QN (Dueling Double Deep Q Network)"""
    
    def __init__(self, net_dims: list, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mid_dim = net_dims[0] if net_dims else 128
        
        # State normalization parameters
        self.state_avg = torch.zeros(state_dim)
        self.state_std = torch.ones(state_dim)
        
        # Value normalization parameters
        self.value_avg = torch.zeros(1)
        self.value_std = torch.ones(1)
        
        # Shared feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.ReLU()
        )
        
        # Value streams for both networks
        self.value1_net = nn.Sequential(
            nn.Linear(self.mid_dim, self.mid_dim // 2),
            nn.ReLU(),
            nn.Linear(self.mid_dim // 2, 1)
        )
        self.value2_net = nn.Sequential(
            nn.Linear(self.mid_dim, self.mid_dim // 2),
            nn.ReLU(),
            nn.Linear(self.mid_dim // 2, 1)
        )
        
        # Advantage streams for both networks
        self.advantage1_net = nn.Sequential(
            nn.Linear(self.mid_dim, self.mid_dim // 2),
            nn.ReLU(),
            nn.Linear(self.mid_dim // 2, action_dim)
        )
        self.advantage2_net = nn.Sequential(
            nn.Linear(self.mid_dim, self.mid_dim // 2),
            nn.ReLU(),
            nn.Linear(self.mid_dim // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both dueling Q-networks
        Args:
            state: Input state tensor of shape (batch_size, state_dim)
        Returns:
            q1: Q-values from first dueling network
            q2: Q-values from second dueling network
        """
        features = self.feature_net(state)
        
        # First dueling network
        value1 = self.value1_net(features)
        advantage1 = self.advantage1_net(features)
        q1 = value1 + advantage1 - advantage1.mean(dim=1, keepdim=True)
        
        # Second dueling network
        value2 = self.value2_net(features)
        advantage2 = self.advantage2_net(features)
        q2 = value2 + advantage2 - advantage2.mean(dim=1, keepdim=True)
        
        return q1, q2
    
    def get_q_min(self, state: torch.Tensor) -> torch.Tensor:
        """Get minimum Q-value between the two dueling networks"""
        q1, q2 = self.forward(state)
        return torch.min(q1, q2)
    
    def get_q_max(self, state: torch.Tensor) -> torch.Tensor:
        """Get maximum Q-value between the two dueling networks"""
        q1, q2 = self.forward(state)
        return torch.max(q1, q2)
    
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get action with highest Q-value"""
        q1, q2 = self.forward(state)
        q_values = torch.min(q1, q2)  # Conservative estimation
        return q_values.argmax(dim=1, keepdim=True)
    
    def get_q1_q2(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q1 and Q2 values separately"""
        return self.forward(state)


class QNet(nn.Module):
    """Standard Q-Network for DQN"""
    
    def __init__(self, net_dims: list, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mid_dim = net_dims[0] if net_dims else 128
        
        # State normalization parameters
        self.state_avg = torch.zeros(state_dim)
        self.state_std = torch.ones(state_dim)
        
        # Value normalization parameters
        self.value_avg = torch.zeros(1)
        self.value_std = torch.ones(1)
        
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network
        Args:
            state: Input state tensor of shape (batch_size, state_dim)
        Returns:
            q_values: Q-values for all actions
        """
        return self.q_net(state)
    
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get action with highest Q-value"""
        q_values = self.forward(state)
        return q_values.argmax(dim=1, keepdim=True)
    
    def get_q1_q2(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q1 and Q2 values separately (for compatibility with twin networks)"""
        q_values = self.forward(state)
        return q_values, q_values  # Return same values for both Q1 and Q2