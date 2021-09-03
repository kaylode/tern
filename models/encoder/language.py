import torch
import torch.nn as nn

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

    def __init__(self, version='distilbert-base-uncased'):
        super().__init__()

        model = AutoModel.from_pretrained(version)

        tokenizer = AutoTokenizer.from_pretrained(
            version, add_special_tokens = 'true', padding = 'longest')

        self.pipeline = pipeline(
            'feature-extraction', 
            model=model, 
            tokenizer=tokenizer, 
            device = 0)

    def forward(self, inputs):
        with torch.no_grad():
            outputs = self.pipeline(inputs)
        return np.array(outputs, dtype=np.float32)