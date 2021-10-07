import torch
import torch.nn as nn
from datasets.utils import make_feature_batch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline

class EncoderBERT(nn.Module):
    """
    Pretrained BERT model
    :input:
        version:       bert version
    :output:
        encoded embeddings shape [batch * input length * model_dim]
    """

    def __init__(self, version='distilbert-base-uncased', precomp=True, device=None):
        super().__init__()

        self.precomp = precomp

        if not self.precomp:
            model = AutoModel.from_pretrained(version)

            tokenizer = AutoTokenizer.from_pretrained(
                version, add_special_tokens = 'true', padding = 'longest')

            self.pipeline = pipeline(
                'feature-extraction', 
                model=model, 
                tokenizer=tokenizer, 
                device = 0 if device is not None else -1)

            self.device = device
            
    def forward(self, x):
        if not self.precomp:
            with torch.no_grad():
                x = self.pipeline(x)
            x = np.squeeze(x)
            if len(x.shape) == 2:
                x = np.expand_dims(x, axis=0)
            x = make_feature_batch(x, pad_token=0)
            x = x.to(self.device)

        return x