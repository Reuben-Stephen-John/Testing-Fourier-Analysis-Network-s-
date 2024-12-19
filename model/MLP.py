import torch 
import torch.nn as nn

class MLP (nn.Module):
    def __init__(self, in_features, hidden_dim, num_layers):
        super().__init__()

        self.layers= nn.ModuleList()

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(in_features, hidden_dim))
            self.layers.append(nn.GELU())
            in_features = hidden_dim

        # last layer
        self.layers.append(nn.Linear(hidden_dim,1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x