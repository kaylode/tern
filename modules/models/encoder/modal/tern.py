from modules.models.encoder import EncoderBottomUp, EncoderBERT
from modules.models.encoder.projection import TransformerEncoder, ModalProjection
from modules.models.encoder.utils import init_xavier, l2norm
from .base import CrossModal

class TERN(CrossModal):
    """
    Architecture idea based on Transformer Encoder Reasoning Network
    """
    def __init__(self, d_model, d_embed, d_ff, N_v, N_l, heads, dropout, aggregation, precomp_bert, num_sw_layers, **kwargs):
        super(TERN, self).__init__()
        self.name = "TERN"
        self.aggregation = aggregation
        self.precomp_bert = precomp_bert

        self.encoder_v = EncoderBottomUp(feat_dim=2048, d_model=d_model)
        self.encoder_l = EncoderBERT(precomp=precomp_bert)

        self.reasoning_v = TransformerEncoder(d_model=d_model, d_ff=d_ff, N=N_v, heads=heads, dropout=dropout)
        self.reasoning_l = TransformerEncoder(d_model=d_model, d_ff=d_ff, N=N_l, heads=heads, dropout=dropout)
        
        self.img_proj = ModalProjection(in_dim=d_model, out_dim=d_embed)
        self.cap_proj = ModalProjection(in_dim=d_model, out_dim=d_embed)

        # Shared weight encoders
        if num_sw_layers > 0:
            self.sw_layer = TransformerEncoder(d_model=d_model, d_ff=d_ff, N=num_sw_layers, heads=heads, dropout=dropout)
        else:
            self.sw_layer = None

        init_xavier(self)

    def forward(self, batch, device):
        outputs_v = self.visual_forward(batch, device)
        outputs_l = self.lang_forward(batch, device)
        return outputs_l, outputs_v
        

    def visual_forward(self, batch, device):
        visual_inputs = batch['feats'].to(device)
        spatial_inputs = batch['loc_feats'].to(device)

        outputs_v = self.encoder_v(visual_inputs, spatial_inputs) 
        outputs_v = self.reasoning_v(outputs_v)                    #[B x 37 x d_model] (append CLS token to first)

        if self.sw_layer is not None:
            outputs_v = self.sw_layer(outputs_v)

        if self.aggregation == 'mean':
            feats_v = self.img_proj(outputs_v).mean(dim=1)
        if self.aggregation == 'first':
            feats_v = self.img_proj(outputs_v)[:, 0]

        feats_v = l2norm(feats_v)
        return feats_v

    def lang_forward(self, batch, device):
        if self.precomp_bert:
            lang_inputs = batch['lang_feats'].to(device)
            outputs_l = self.encoder_l(lang_inputs) 
        else:
            outputs_l = self.encoder_l(batch['texts'])
            outputs_l = outputs_l.to(device)

        outputs_l = self.reasoning_l(outputs_l) #[B x Length+2 x d_model] (plus 2 special tokens)
        
        if self.sw_layer is not None:
            outputs_l = self.sw_layer(outputs_l)

        if self.aggregation == 'mean':
            feats_l = self.cap_proj(outputs_l).mean(dim=1)
        
        if self.aggregation == 'first':
            feats_l = self.cap_proj(outputs_l)[:, 0]

        feats_l = l2norm(feats_l)
        return feats_l