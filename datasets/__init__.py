from .vocab import Vocabulary
from .cocoset import CocoDataset, NumpyFeatureDataset
from .dataloader import NumpyFeatureLoader, RawNumpyFeatureLoader
from .utils import make_feature_batch

klass = globals()["class_name"]
instance = klass()

def get_loader(config, device):
    config['device'] = device
    dataloader = globals()[config['name']](**config)
    return dataloader.dataset, dataloader