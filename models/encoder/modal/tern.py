from models.encoder import EncoderBottomUp, EncoderBERT, TransformerEncoder, ModalProjection
from models.encoder.utils import init_xavier, l2norm
from .base import CrossModal

class TERN(CrossModal):
    """
    Architecture idea based on Transformer Encoder Reasoning Network
    """
    def __init__(self, config, precomp_bert=True):
        super(TERN, self).__init__()
        self.name = config["name"]
        self.aggregation = config["aggregation"]

        self.encoder_v = EncoderBottomUp(feat_dim=2048, d_model=config['d_model'])
        self.encoder_l = EncoderBERT(precomp=precomp_bert)

        self.reasoning_v = TransformerEncoder(d_model=config['d_model'], d_ff=config["d_ff"], N=config["N_v"], heads=config["heads"], dropout=config["dropout"])
        self.reasoning_l = TransformerEncoder(d_model=config['d_model'], d_ff=config["d_ff"], N=config["N_l"], heads=config["heads"], dropout=config["dropout"])
        
        self.img_proj = ModalProjection(in_dim=config['d_model'], out_dim=config["d_embed"])
        self.cap_proj = ModalProjection(in_dim=config['d_model'], out_dim=config["d_embed"])

        # Shared weight encoders
        # self.transformer_encoder = TransformerEncoder(d_model=1024, d_ff=2048, N=2, heads=4, dropout=0.1)
        init_xavier(self)

    def forward_batch(self, batch, device):
        src_inputs = batch['feats'].to(device)
        loc_src_inputs = batch['loc_feats'].to(device)
        lang_src_inputs = batch['lang_feats'].to(device)

        outputs_l, outputs_v = self.forward(
            visual_inputs=src_inputs, 
            spatial_inputs=loc_src_inputs, 
            lang_inputs=lang_src_inputs)

        return outputs_l, outputs_v
        
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