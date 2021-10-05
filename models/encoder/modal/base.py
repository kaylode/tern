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
        self.aggregation = None

    