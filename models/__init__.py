import torch.nn as nn
from models.encoder import EncoderBottomUp, EncoderBERT, TransformerEncoder, ModalProjection
from models.encoder.utils import init_xavier, l2norm
from .retriever import Retriever

from models.encoder.modal import get_cross_modal