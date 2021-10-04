import timm
import torch.nn as nn

TIMM_MODELS = [
        "deit_tiny_distilled_patch16_224", 
        'deit_small_distilled_patch16_224', 
        'deit_base_distilled_patch16_224',
        'deit_base_distilled_patch16_384']

def get_pretrained_encoder(model_name):
    assert model_name in TIMM_MODELS, "Timm Model not found"
    model = timm.create_model(model_name, pretrained=True)
    return model

class EncoderVIT(nn.Module):
    """
    Pretrained Transformers Encoder from timm Vision Transformers
    :output:
        encoded embeddings shape [batch * (image_size/patch_size)**2 * model_dim]
    """
    def __init__(self, model_name='deit_base_distilled_patch16_224'):
        super().__init__()
        
        vit = get_pretrained_encoder(model_name)
        self.embed_dim = vit.embed_dim 
        self.patch_embed = vit.patch_embed
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        
    def forward(self, src):
        x = self.patch_embed(src)
        x = self.pos_drop(x + self.pos_embed[:, 2:]) # skip dis+cls tokens
        x = self.blocks(x)
        x = self.norm(x)
        return x