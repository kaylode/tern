import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses 

class TripletLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.loss_fn = losses.TripletMarginLoss(**kwargs)

    def forward(self, feats1, feats2):
        labels = torch.arange(feats1.size(0))
        embeddings = torch.cat([feats1, feats2], dim=0)
        labels = torch.cat([labels, labels], dim=0)

        loss = self.loss_fn(embeddings, labels)
        return {'T': loss}