from models.encoder.modal.vitr import ViTR
from .tern import TERN
from .vitr import ViTR

def get_cross_modal(config):
    if config['name'] == 'TERN':
        return TERN(config)

    if config['name'] == 'ViTR':
        return ViTR(config)