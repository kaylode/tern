import copy
from posix import XATTR_REPLACE
import torch
import torch.nn as nn

import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from models.transformer import EncoderLayer, LayerNorm

def get_clones(module, N):
    """
    "Produce N identical layers."
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class EncoderBERT(nn.Module):
    """
    Pretrained BERT model
    :input:
        version:       bert version
    :output:
        encoded embeddings shape [batch * input length * model_dim]
    """

    def __init__(self, d_model, d_ff, N, heads, dropout, version='distilbert-base-uncased', precomp=True):
        super().__init__()

        self.N = N
        self.precomp = precomp

        if not self.precomp:
            model = AutoModel.from_pretrained(version)

            tokenizer = AutoTokenizer.from_pretrained(
                version, add_special_tokens = 'true', padding = 'longest')

            self.pipeline = pipeline(
                'feature-extraction', 
                model=model, 
                tokenizer=tokenizer, 
                device = 0)
        
        if self.N > 0:
            self.layers = get_clones(EncoderLayer(d_model, d_ff, heads, dropout), N)
            self.norm = LayerNorm(d_model)   

    def forward(self, x):
        if not self.precomp:
            with torch.no_grad():
                x = self.pipeline(x)

        if self.N > 0:
            for i in range(self.N):
                x = self.layers[i](x, mask=None)
            x = self.norm(x)
        return x