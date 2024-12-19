import torch
import torch.nn as nn

class FANlayer(nn.Module):
    def __init__(self, in_features, d_p, d_p_bar, activation=nn.GELU()):
        super().__init__()
        self.Wp= nn.Parameter(torch.randn(in_features,d_p))
        self.Wp_bar=nn.Parameter(torch.randn(in_features,d_p_bar))
        self.Bp_bar=nn.Parameter(torch.zeros(d_p_bar))
        self.activation=activation

    def forward(self, x):
        cos_term = torch.cos(torch.matmul(x,self.Wp))
        sin_term = torch.sin(torch.matmul(x,self.Wp))
        non_periodic_term = self.activation(torch.matmul(x,self.Wp_bar)+self.Bp_bar)
        return torch.cat([cos_term,sin_term,non_periodic_term],dim=-1)
    
class FAN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_layers, activation = nn.GELU()):
        super().__init__()

        self.num_layers = num_layers
        self.layers= nn.ModuleList()

        d_p = hidden_dim//4
        d_p_bar = hidden_dim

        for _ in range(num_layers - 1):
            self.layers.append(FANlayer(in_features,d_p,d_p_bar,activation))
            in_features = 2 * d_p + d_p_bar

        self.WL = nn.Parameter(torch.randn(in_features,1))
        self.BL = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.matmul(x, self.WL) + self.BL

    