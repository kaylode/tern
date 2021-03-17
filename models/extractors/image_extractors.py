import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

from .base_extractor import ExtractorNetwork
from efficientnet_pytorch import EfficientNet

def _get_cnn_basemodel(cnn_model_name):
    model_name, version = cnn_model_name.split('_')

    if "efficientnet" in cnn_model_name:
        return EfficientNetExtractor(version=version)

    if "resnet" in cnn_model_name:
        return ResNetExtractor(version=version)
    
class EfficientNetExtractor(ExtractorNetwork):
    def __init__(self, version, freeze=False):
        super().__init__()
        self.extractor = EfficientNet.from_pretrained(
            f'efficientnet-{version}')
        self.feature_dim = self.extractor._fc.in_features

        if freeze:
            self.freeze()

    def forward(self, x):
        x = self.extractor.extract_features(x)
        x = self.extractor._avg_pooling(x)
        x = x.view(x.size(0), -1)
        return x


class ResNetExtractor(ExtractorNetwork):
    arch = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(self, version):
        super().__init__()
        assert version in [18, 34, 50, 101, 152], \
            'ResNet{version} is not implemented.'
        cnn = ResNetExtractor.arch[version](pretrained=False)
        self.extractor = nn.Sequential(*list(cnn.children())[:-1])
        self.feature_dim = cnn.fc.in_features

    def forward(self, x):
        return self.extractor(x).view(x.size(0), -1)