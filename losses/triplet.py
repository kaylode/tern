import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses 

class TripletLoss(losses.TripletMarginLoss):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, feats1, feats2):
        labels = torch.arange(feats1.size(0))
        embeddings = torch.cat([feats1, feats2], dim=0)
        labels = torch.cat([labels, labels], dim=0)

        loss = self(embeddings, labels)
        return {'T': loss}