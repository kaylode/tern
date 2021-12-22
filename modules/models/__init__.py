import torch.nn as nn
from .encoder import ViTR, TERN, SGRAF, CLIP, EncoderBERT
from .encoder.utils import init_xavier, l2norm
from .retriever import Retriever
