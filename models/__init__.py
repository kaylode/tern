import torch.nn as nn
from models.encoder import ViTR, TERN, SGRAF, CLIP
from models.encoder.utils import init_xavier, l2norm
from .retriever import Retriever
