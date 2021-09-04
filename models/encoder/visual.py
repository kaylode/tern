import torch
import torch.nn as nn
import copy

from models.transformer import FeatureEmbedding, SpatialEncoding, EncoderLayer, LayerNorm

def get_clones(module, N):
    """
    "Produce N identical layers."
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class EncoderBottomUp(nn.Module):
    """
    Encoder for Bottom-Up-Attention features
    :input:
        feat_dim:       feature dim
        d_model:        embeddings dim
        d_ff:           feed-forward dim
        N:              number of layers
        heads:          number of attetion heads
        dropout:        dropout rate
    :output:
        encoded embeddings shape [batch * input length * model_dim]
    """
    def __init__(self, feat_dim, d_model, d_ff, N, heads, dropout):
        super().__init__()
        self.N = N
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, d_model)) # Add CLS token 
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, d_model)) # Add CLS token 

        self.feat_embed = FeatureEmbedding(feat_dim, d_model)
        self.loc_embed = SpatialEncoding(d_model)
        self.layers = get_clones(EncoderLayer(d_model, d_ff, heads, dropout), N)
        self.norm = LayerNorm(d_model)    
    def forward(self, src, spatial_src):
        x = self.feat_embed(src)
        cls_token1 = self.cls_token1.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token1, x), dim=1)

        spatial_x = self.loc_embed(spatial_src)
        cls_token2 = self.cls_token2.expand(spatial_x.shape[0], -1, -1)
        spatial_x = torch.cat((cls_token2, spatial_x), dim=1)

        x += spatial_x

        for i in range(self.N):
            x = self.layers[i](x, mask=None)
        x = self.norm(x)
        return x