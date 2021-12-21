import torch.nn as nn


class CrossModal(nn.Module):
    """
    Abstract class for CrossModal
    :output:
        features of inputs
    """
    def __init__(self, **kwargs):
        super(CrossModal, self).__init__()
        self.name = None

    def forward(self, batch, device):
        raise NotImplementedError("Abtract class method is not implemented")

    def visual_forward(self, batch, device):
        raise NotImplementedError("Abtract class method is not implemented")

    def lang_forward(self, batch, device):
        raise NotImplementedError("Abtract class method is not implemented")
