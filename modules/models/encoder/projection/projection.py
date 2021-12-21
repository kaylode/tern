import torch.nn as nn

from models.encoder.utils import get_clones
from models.modules.transformer import EncoderLayer, LayerNorm

class TransformerEncoder(nn.Module):
    """
    Shared weight transformer encoder
    """
    def __init__(self, d_model, d_ff, N, heads, dropout):
        super().__init__()
        self.N = N

        if self.N != 0:
            self.layers = get_clones(EncoderLayer(d_model, d_ff, heads, dropout), N)
            self.norm = LayerNorm(d_model)  

    def forward(self, x):
        if self.N != 0:
            for i in range(self.N):
                x = self.layers[i](x, mask=None)
            x = self.norm(x)
        return x

class ModalProjection(nn.Module):
    """
    Project features into same space
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.model = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.model(x)