import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

from .base_extractor import ExtractorNetwork
from transformers import AutoModel

def _get_bert_basemodel(bert_model_name, freeze_layers=None):
    return BertExtractor(bert_model_name, freeze_layers)

class BertExtractor(ExtractorNetwork):
    """Baseline model"""

    def __init__(self, version='bert-base-uncased', freeze_layers=None):
        super().__init__()

        self.extractor = AutoModel.from_pretrained(version)
        self.feature_dim = self.extractor.config.hidden_size

        if freeze_layers is not None:
            self.freeze_some_layers(freeze_layers)

    def freeze_some_layers(self, freeze_layers=[]):
        for layer_idx in freeze_layers:
            for param in list(self.extractor.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False

    def freeze(self):
        for p in self.extractor.parameters():
            p.requires_grad = False

    def forward(self, encoded_inputs):
        outputs = self.extractor(**encoded_inputs)
        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask'])

        return sentence_embeddings

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask