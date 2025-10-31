import torch as th
import torch.nn as nn
from typing import Optional, Tuple

TEN = th.Tensor


class RnnRegNet(nn.Module):
    def __init__(self, inp_dim: int, mid_dim: int, out_dim: int, num_layers: int = 4):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=inp_dim,
            hidden_size=mid_dim,
            num_layers=num_layers,
            batch_first=False,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=mid_dim,
            hidden_size=mid_dim,
            num_layers=num_layers,
            batch_first=False,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(mid_dim, mid_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mid_dim // 2, out_dim)
        )
        
    def forward(self, inp: TEN, hid: Optional[Tuple[TEN, TEN]] = None) -> Tuple[TEN, Tuple[TEN, TEN]]:
        """
        Forward pass through the network
        Args:
            inp: Input tensor of shape (seq_len, batch_size, inp_dim)
            hid: Hidden state tuple (h, c) from previous forward pass
        Returns:
            out: Output tensor of shape (seq_len, batch_size, out_dim)
            hid: Updated hidden state tuple
        """
        # LSTM forward pass
        lstm_out, (h_lstm, c_lstm) = self.lstm(inp, hid)
        
        # GRU forward pass
        gru_out, h_gru = self.gru(lstm_out, h_lstm)
        
        # Output projection
        out = self.out_proj(gru_out)
        
        # Return output and hidden state
        hid = (h_gru, c_lstm)
        return out, hid
    
    def get_obj_value(self, criterion, out: TEN, lab: TEN, wup_dim: int) -> TEN:
        """
        Calculate objective value (loss) for training
        Args:
            criterion: Loss function
            out: Model output
            lab: Target labels
            wup_dim: Warm-up dimension (skip first wup_dim steps)
        Returns:
            obj: Objective value tensor
        """
        seq_len = min(out.shape[0], lab.shape[0])
        out_trimmed = out[wup_dim:seq_len, :, :]
        lab_trimmed = lab[wup_dim:seq_len, :, :]
        
        obj = criterion(out_trimmed, lab_trimmed)
        return obj