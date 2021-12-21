from modules.models.encoder import EncoderVIT, EncoderBERT
from modules.models.encoder.projection import ModalProjection
from modules.models.encoder.utils import init_xavier, l2norm
from .base import CrossModal
import torch.nn as nn

class ViTR(CrossModal):
    """
    Architecture idea based on Vision Transformer Retrieval
    """
    def __init__(self, d_model, d_embed, aggregation, precomp_bert, device, **kwargs):
        super(ViTR, self).__init__()
        self.name = "ViTR"
        self.aggregation = aggregation

        self.encoder_v = EncoderVIT()
        self.encoder_l = EncoderBERT(precomp=precomp_bert, device=device)
        
        self.img_proj = ModalProjection(in_dim=d_model, out_dim=d_embed)
        self.cap_proj = ModalProjection(in_dim=d_model, out_dim=d_embed)

        # Shared weight encoders
        # self.transformer_encoder = TransformerEncoder(d_model=1024, d_ff=2048, N=2, heads=4, dropout=0.1)
        init_xavier(self)

    def forward_batch(self, batch, device):
        visual_inputs = batch['imgs'].to(device)
        lang_inputs = batch['texts']

        outputs_l, outputs_v = self.forward(
            visual_inputs=visual_inputs, 
            lang_inputs=lang_inputs)

        return outputs_l, outputs_v

    def forward(self, visual_inputs, lang_inputs):
        feats_v = self.visual_forward(visual_inputs)
        feats_l = self.lang_forward(lang_inputs)

        return feats_l, feats_v

    def visual_forward(self, visual_inputs):
        outputs_v = self.encoder_v(visual_inputs)                   #[B x 196 x d_model]

        if self.aggregation == 'mean':
            feats_v = self.img_proj(outputs_v).mean(dim=1)
        if self.aggregation == 'first':
            feats_v = self.img_proj(outputs_v)[:, 0]

        feats_v = l2norm(feats_v)
        return feats_v

    def lang_forward(self, lang_inputs):
        outputs_l = self.encoder_l(lang_inputs)                     #[B x Length+2 x d_model] (plus 2 special tokens)

        if self.aggregation == 'mean':
            feats_l = self.cap_proj(outputs_l).mean(dim=1)
        
        if self.aggregation == 'first':
            feats_l = self.cap_proj(outputs_l)[:, 0]

        feats_l = l2norm(feats_l)
        return feats_l