import torch.nn as nn
from .encoder import ViTR, TERN, SGRAF, CLIP
from .encoder.utils import init_xavier, l2norm
from .retriever import Retriever
