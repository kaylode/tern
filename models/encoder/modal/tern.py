from models.encoder import EncoderBottomUp, EncoderBERT, TransformerEncoder, ModalProjection
from models.encoder.utils import init_xavier
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