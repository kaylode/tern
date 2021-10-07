import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses 

class CustomNTXentLoss(nn.Module):
    # Source: https://arxiv.org/pdf/2010.00747.pdf
    def __init__(self, temperature=1., weight=0.5):
        super(CustomNTXentLoss, self).__init__()
        self.temperature = temperature
        self.weight = weight
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, img_features, txt_features):
        v = img_features
        u = txt_features

        similarity_vu = self.cossim(v.unsqueeze(-2), u.unsqueeze(-3)) / self.temperature
        similarity_uv = self.cossim(u.unsqueeze(-2), v.unsqueeze(-3)) / self.temperature
        
        dim = similarity_vu.shape[-1]
        
        targets = torch.arange(dim, dtype=torch.long, device=similarity_vu.device)
        
        loss_vu = self.ce(similarity_vu, targets)
        loss_uv = self.ce(similarity_uv, targets)

        loss_vu = loss_vu.mean()
        loss_uv = loss_uv.mean()
        
        total_loss = self.weight*loss_vu + (1-self.weight)*loss_uv

        return {
            'T': total_loss, 
            'I-T': loss_vu,
            'T-I': loss_uv
        }


class NTXentLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.loss_fn = losses.NTXentLoss(**kwargs)

    def forward(self, feats1, feats2):
        labels = torch.arange(feats1.size(0))
        embeddings = torch.cat([feats1, feats2], dim=0)
        labels = torch.cat([labels, labels], dim=0)

        loss = self.loss_fn(embeddings, labels)
        return {'T': loss}