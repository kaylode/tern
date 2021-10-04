import torch.nn as nn
from models.encoder import EncoderBottomUp, EncoderBERT, TransformerEncoder, ModalProjection
from models.encoder.utils import init_xavier, l2norm

class CrossModal(nn.Module):
    """
    Abstract class for CrossModal
    :output:
        features of inputs
    """
    def __init__(self, **kwargs):
        super(CrossModal, self).__init__()
        self.name = None
        self.aggregation = None

    def forward(self, visual_inputs, spatial_inputs, lang_inputs):
        feats_v = self.visual_forward(visual_inputs, spatial_inputs)
        feats_l = self.lang_forward(lang_inputs)

        return feats_l, feats_v

    def visual_forward(self, visual_inputs, spatial_inputs):
        outputs_v = self.encoder_v(visual_inputs, spatial_inputs) 
        outputs_v = self.reasoning_v(outputs_v)                    #[B x 37 x d_model] (append CLS token to first)

        if self.aggregation == 'mean':
            feats_v = self.img_proj(outputs_v).mean(dim=1)
        if self.aggregation == 'first':
            feats_v = self.img_proj(outputs_v)[:, 0]

        feats_v = l2norm(feats_v)
        return feats_v

    def lang_forward(self, lang_inputs):
        outputs_l = self.encoder_l(lang_inputs) 
        outputs_l = self.reasoning_l(outputs_l) #[B x Length+2 x d_model] (plus 2 special tokens)

        if self.aggregation == 'mean':
            feats_l = self.cap_proj(outputs_l).mean(dim=1)
        
        if self.aggregation == 'first':
            feats_l = self.cap_proj(outputs_l)[:, 0]

        feats_l = l2norm(feats_l)
        return feats_l