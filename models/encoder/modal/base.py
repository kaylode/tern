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

    def forward_batch(self, batch, device):
        raise NotImplementedError("Abtract class method is not implemented")

    def forward(self, visual_inputs, spatial_inputs, lang_inputs):
        raise NotImplementedError("Abtract class method is not implemented")

    def visual_forward(self, visual_inputs, spatial_inputs):
        raise NotImplementedError("Abtract class method is not implemented")

    def lang_forward(self, lang_inputs):
        raise NotImplementedError("Abtract class method is not implemented")

    def visual_forward_batch(self, batch, device):
        raise NotImplementedError("Abtract class method is not implemented")

    def lang_forward_batch(self, batch, device):
        raise NotImplementedError("Abtract class method is not implemented")
