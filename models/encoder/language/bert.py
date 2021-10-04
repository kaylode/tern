import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel, pipeline

class EncoderBERT(nn.Module):
    """
    Pretrained BERT model
    :input:
        version:       bert version
    :output:
        encoded embeddings shape [batch * input length * model_dim]
    """

    def __init__(self, version='distilbert-base-uncased', precomp=True):
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
                device = 0)
        

    def forward(self, x):
        if not self.precomp:
            with torch.no_grad():
                x = self.pipeline(x)

        return x