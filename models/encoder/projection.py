import torch
import torch.nn as nn

from .visual import get_clones, EncoderLayer, LayerNorm

class TransformerEncoder(nn.Module):
    """
    Shared weight transformer encoder
    """
    def __init__(self, d_model, d_ff, N, heads, dropout):
        self.layers = get_clones(EncoderLayer(d_model, d_ff, heads, dropout), N)
        self.norm = LayerNorm(d_model)  

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x, mask=None)
        x = self.norm(x)
        return x[:, 0, :]

