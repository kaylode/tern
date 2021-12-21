import torch
import torch.nn as nn
from models.encoder.projection import FeatureEmbedding, SpatialEncoding

class EncoderBottomUp(nn.Module):
    """
    Encoder for Bottom-Up-Attention features
    :input:
        feat_dim:       feature dim
        d_model:        embeddings dim
    :output:
        encoded embeddings shape [batch * input length * model_dim]
    """
    def __init__(self, feat_dim, d_model):
        super().__init__()
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, d_model)) # Add CLS token 
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, d_model)) # Add CLS token 

        self.feat_embed = FeatureEmbedding(feat_dim, d_model)
        self.loc_embed = SpatialEncoding(d_model)
 
    def forward(self, src, spatial_src):
        x = self.feat_embed(src)
        cls_token1 = self.cls_token1.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token1, x), dim=1)

        spatial_x = self.loc_embed(spatial_src)
        cls_token2 = self.cls_token2.expand(spatial_x.shape[0], -1, -1)
        spatial_x = torch.cat((cls_token2, spatial_x), dim=1)

        x += spatial_x

        return x