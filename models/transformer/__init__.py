import torch.nn as nn
from models.encoder import EncoderBottomUp, EncoderBERT
from .utils import init_xavier

class TransformerBottomUp(nn.Module):
    """
    Transformer model
    :input:
        patch_size:    size of patch
        trg_vocab:     size of target vocab
        d_model:       embeddings dim
        d_ff:          feed-forward dim
        N:             number of layers
        heads:         number of attetion heads
        dropout:       dropout rate
    :output:
        next words probability shape [batch * input length * vocab_dim]
    """
    def __init__(self, feat_dim, d_model=1024, d_ff=3072, N_enc=4, heads=4, dropout=0.2):
        super().__init__()
        self.name = "Transformer"

        self.encoder_v = EncoderBottomUp(feat_dim, d_model, d_ff, N_enc, heads, dropout)
        self.encoder_l = EncoderBERT()
        
        init_xavier(self)

    def forward(self, visual_inputs, spatial_inputs, lang_inputs):
        outputs_v = self.encoder_v(visual_inputs, spatial_inputs)
        outputs_l = self.encoder_l(lang_inputs)

        return outputs_v