import torch
from .nt_xent import NTXentLoss, CustomNTXentLoss
from .contrastive import ContrastiveLoss
from .triplet import TripletLoss

from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity, LpDistance
from pytorch_metric_learning.reducers import ThresholdReducer, MeanReducer
from pytorch_metric_learning.regularizers import LpRegularizer

def get_distance(name):
    if name == 'cosine':
        return CosineSimilarity()
    if name == 'dot':
        return DotProductSimilarity()
    if name == 'euclidean':
        return LpDistance()

    raise ValueError("Loss function not exists")

def get_reducer(name, **kwargs):
    if name == 'threshold':
        return ThresholdReducer(**kwargs)
    
    raise ValueError("Reducer function not exists")

def get_regularizer(name, **kwargs):
    if name == 'l1':
        return LpRegularizer(p=1, **kwargs)
    if name == 'l2':
        return LpRegularizer(p=2, **kwargs)
    if name == 'mean':
        return MeanReducer(**kwargs)
    
    raise ValueError("Reducer function not exists")

def get_loss_fn(config):

    """
    Metric learning
    https://github.com/KevinMusgrave/pytorch-metric-learning
    """

    kwargs = {}
    if 'distance' in config.keys():
        kwargs['distance'] = get_distance(**config['distance'])
    
    if 'reducer' in config.keys():
        kwargs['reducer'] = get_reducer(**config['reducer'])

    if 'regularizer' in config.keys():
        kwargs['embedding_regularizer'] = get_regularizer(**config['regularizer'])

    loss_fn = None
    if config['name'] == 'triplet':
        loss_fn = TripletLoss

    if config['name'] == 'nxtent':
        loss_fn = NTXentLoss

    if config['name'] == 'custom_nxtent':
        loss_fn = CustomNTXentLoss(temperature=config['temperature'])
        return loss_fn

    if config['name'] == 'contrastive':
        loss_fn = ContrastiveLoss

    if loss_fn is None:
        raise ValueError("Loss function not exists")

    loss_fn = loss_fn(**kwargs)

    return loss_fn