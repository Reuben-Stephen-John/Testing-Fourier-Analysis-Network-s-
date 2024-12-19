import torch
import torch.nn as nn

class FANLayerGated(nn.Module):
    def __init__(self, in_features, d_p, d_p_bar, activation=nn.GELU(), gated=True):
        super().__init__()
        self.Wp = nn.Parameter(torch.randn(in_features, d_p))
        self.Wp_bar = nn.Parameter(torch.randn(in_features, d_p_bar))
        self.Bp_bar = nn.Parameter(torch.zeros(d_p_bar))
        self.activation = activation
        if gated:
            self.gate = nn.Parameter(torch.randn(1))

    def forward(self, x):
        cos_term = torch.cos(torch.matmul(x, self.Wp))
        sin_term = torch.sin(torch.matmul(x, self.Wp))
        non_periodic_term = self.activation(torch.matmul(x, self.Wp_bar) + self.Bp_bar)
        
        if hasattr(self, 'gate'):
            gate = torch.sigmoid(self.gate)
            cos_term = gate * cos_term
            sin_term = gate * sin_term
            non_periodic_term = (1 - gate) * non_periodic_term

        return torch.cat([cos_term, sin_term, non_periodic_term], dim=-1)

# The Gated FAN model
class FANGated(nn.Module):
    def __init__(self, in_features, hidden_dim, num_layers, activation=nn.GELU(), gated=True):
        super().__init__()
        self.layers = nn.ModuleList()
        
        d_p = hidden_dim // 4
        d_p_bar = hidden_dim

        for _ in range(num_layers - 1):
            self.layers.append(FANLayerGated(in_features, d_p, d_p_bar, activation, gated))
            in_features = 2 * d_p + d_p_bar  

        self.output_layer = nn.Linear(in_features, 1) 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)