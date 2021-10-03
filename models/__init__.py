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
    def __init__(self, config, precomp_bert=True):
        super(TERN, self).__init__()
        self.name = config["name"]
        self.aggregation = config["aggregation"]

        self.encoder_v = EncoderBottomUp(feat_dim=2048, d_model=config['d_model'], d_ff=config["d_ff"], N=config["N_v"], heads=config["heads"], dropout=config["dropout"])
        self.encoder_l = EncoderBERT(precomp=precomp_bert, d_model=config['d_model'], d_ff=config["d_ff"], N=config["N_l"], heads=config["heads"], dropout=config["dropout"])
        
        self.img_proj = ModalProjection(in_dim=config['d_model'], out_dim=config["d_embed"])
        self.cap_proj = ModalProjection(in_dim=config['d_model'], out_dim=config["d_embed"])

        # Shared weight encoders
        # self.transformer_encoder = TransformerEncoder(d_model=1024, d_ff=2048, N=2, heads=4, dropout=0.1)
        init_xavier(self)

    def forward(self, visual_inputs, spatial_inputs, lang_inputs):
        feats_v = self.visual_forward(visual_inputs, spatial_inputs)
        feats_l = self.lang_forward(lang_inputs)

        return feats_l, feats_v

    def visual_forward(self, visual_inputs, spatial_inputs):
        outputs_v = self.encoder_v(visual_inputs, spatial_inputs) #[B x 37 x d_model] (append CLS token to first)
        
        if self.aggregation == 'mean':
            feats_v = self.img_proj(outputs_v).mean(dim=1)
        if self.aggregation == 'first':
            feats_v = self.img_proj(outputs_v)[:, 0]

        feats_v = l2norm(feats_v)
        return feats_v

    def lang_forward(self, lang_inputs):
        outputs_l = self.encoder_l(lang_inputs) #[B x Length+2 x d_model] (plus 2 special tokens)
        
        if self.aggregation == 'mean':
            feats_l = self.cap_proj(outputs_l).mean(dim=1)
        
        if self.aggregation == 'first':
            feats_l = self.cap_proj(outputs_l)[:, 0]

        feats_l = l2norm(feats_l)
        return feats_l