import torch.nn as nn
from models.encoder import EncoderBottomUp, EncoderBERT, TransformerEncoder, ModalProjection
from models.transformer import init_xavier, l2norm
from .retriever import Retriever

class TERN(nn.Module):
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
    def __init__(self, embed_dim, precomp_bert=True):
        super(TERN, self).__init__()
        self.name = "TERN"

        self.encoder_v = EncoderBottomUp(feat_dim=2048, d_model=768, d_ff=2048, N=4, heads=2, dropout=0.1)
        self.encoder_l = EncoderBERT(precomp=precomp_bert, d_model=768, d_ff=2048, N=4, heads=2, dropout=0.1)
        
        self.img_proj = ModalProjection(in_dim=768, out_dim=embed_dim)
        self.cap_proj = ModalProjection(in_dim=768, out_dim=embed_dim)

        # Shared weight encoders
        # self.transformer_encoder = TransformerEncoder(d_model=1024, d_ff=2048, N=2, heads=4, dropout=0.1)
        init_xavier(self)

    def forward(self, visual_inputs, spatial_inputs, lang_inputs):
        feats_v = self.visual_forward(visual_inputs, spatial_inputs)
        feats_l = self.lang_forward(lang_inputs)

        return feats_l, feats_v

    def visual_forward(self, visual_inputs, spatial_inputs):
        outputs_v = self.encoder_v(visual_inputs, spatial_inputs) #[B x 37 x d_model] (append CLS token to first)
        feats_v = self.img_proj(outputs_v).mean(dim=1)
        feats_v = l2norm(feats_v)
        return feats_v

    def lang_forward(self, lang_inputs):
        outputs_l = self.encoder_l(lang_inputs) #[B x Length+2 x d_model] (plus 2 special tokens)
        feats_l = self.cap_proj(outputs_l).mean(dim=1)
        feats_l = l2norm(feats_l)
        return feats_l