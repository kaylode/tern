import torch.nn as nn
from models.encoder import EncoderBottomUp, EncoderBERT, TransformerEncoder
from models.transformer import init_xavier, l2norm



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
    def __init__(self, embed_dim, precomp_bert=True):
        super().__init__()
        self.name = "Transformer"

        self.encoder_v = EncoderBottomUp(feat_dim=2048, d_model=768, d_ff=2048, N=4, heads=4, dropout=0.1)
        self.encoder_l = EncoderBERT(precomp=precomp_bert)
        
        self.img_proj = nn.Linear(768, embed_dim)
        self.cap_proj = nn.Linear(768, embed_dim)

        # Shared weight encoders
        self.transformer_encoder = TransformerEncoder(d_model=1024, d_ff=2048, N=2, heads=4, dropout=0.1)
        init_xavier(self)

    def forward(self, visual_inputs, spatial_inputs, lang_inputs):
        outputs_v = self.encoder_v(visual_inputs, spatial_inputs) #[B x 37 x d_model] (append CLS token to first)
        outputs_l = self.encoder_l(lang_inputs) #[B x Length+2 x d_model] (plus 2 special tokens)

        outputs_v = self.img_proj(outputs_v)
        outputs_l = self.cap_proj(outputs_l)

        feats_v = self.transformer_encoder(outputs_v)
        feats_l = self.transformer_encoder(outputs_l)

        feats_l = l2norm(feats_l)
        feats_v = l2norm(feats_v)

        return feats_l, feats_v