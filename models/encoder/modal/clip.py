import clip
from .base import CrossModal

class CLIP(CrossModal):
    """
    CLIP feature extractor by OpenAI
    Only for inference, not training
    """
    def __init__(self, name, **kwargs):
        super(CLIP, self).__init__()
        self.name = "CLIP"

        self.model, _ = clip.load(name)
        self.model.eval()
      
    def forward_batch(self, batch, device):
        imgs = batch['imgs'].to(device)
        texts = batch['texts'].to(device)

        outputs_l, outputs_v = self.forward(
            visual_inputs=imgs, 
            lang_inputs=texts)

        return outputs_l, outputs_v
        
    def forward(self, visual_inputs, lang_inputs):
        feats_l = self.lang_forward(lang_inputs)
        feats_v = self.visual_forward(visual_inputs)
        return feats_l, feats_v

    def visual_forward(self, visual_inputs):
        feats_v = self.model.encode_image(visual_inputs) #[B x 512]    
        return feats_v

    def lang_forward(self, lang_inputs):
        feats_l = self.model.encode_text(lang_inputs) #[B x 512]  
        return feats_l

    def visual_forward_batch(self, batch, device):
        visual_inputs = batch['imgs'].to(device)
        feats_v = self.model.encode_image(visual_inputs) #[B x 512]   
        return feats_v

    def lang_forward_batch(self, batch, device):
        lang_inputs = batch['texts'].to(device)
        feats_l = self.model.encode_text(lang_inputs) #[B x 512]  
        return feats_l