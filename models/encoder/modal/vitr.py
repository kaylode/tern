from models.encoder import EncoderVIT, EncoderBERT, TransformerEncoder, ModalProjection
from models.encoder.utils import init_xavier
from .base import CrossModal
import torch.nn as nn

class ViTR(CrossModal):
    """
    Architecture idea based on Vision Transformer Retrieval
    """
    def __init__(self, config, precomp_bert=True):
        super(ViTR, self).__init__()
        self.name = config["name"]
        self.aggregation = config["aggregation"]

        self.encoder_v = EncoderVIT()
        self.encoder_l = EncoderBERT(precomp=precomp_bert)

        self.reasoning_v = nn.Identity()
        self.reasoning_l = nn.Identity()
        
        self.img_proj = ModalProjection(in_dim=config['d_model'], out_dim=config["d_embed"])
        self.cap_proj = ModalProjection(in_dim=config['d_model'], out_dim=config["d_embed"])

        # Shared weight encoders
        # self.transformer_encoder = TransformerEncoder(d_model=1024, d_ff=2048, N=2, heads=4, dropout=0.1)
        init_xavier(self)