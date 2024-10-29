import torch
import torch.nn as nn
from transformers import SamConfig
from transformers.models.sam.modeling_sam import (
    SamModel, SamVisionEncoder, SamMaskDecoder, SamPositionalEmbedding, SamPromptEncoder
)

_HF_PRETRAIN_NAME = "pretrain/sam-vit-base/"

class GSAMBase(nn.Module):
    def __init__(self, hf_pretrain_name, device):
        super().__init__()
        sam_config = SamConfig.from_pretrained(hf_pretrain_name)
        self.model = SamModel(sam_config).from_pretrained(hf_pretrain_name).to(device)
        self.vision_encoder = self.model.vision_encoder
        self.prompter_encoder = self.model.prompt_encoder
        self.mask_decoder = self.model.mask_decoder
        


class GSAMVisionEncoder(GSAMBase):
    def __init__(self, hf_pretrain_name, device):
        super().__init__(hf_pretrain_name, device)
        self.g_vision_encoder = self.vision_encoder
        self.g_vision_encoder.is_init = True
        self.img_position_embedding = self.model.get_image_wide_positional_embeddings()

    def forward(self, x):
        return self.g_vision_encoder(x)


class GSAMPromptEncoder(GSAMBase):
    def __init__(self, hf_pretrain_name, device):
        super().__init__(hf_pretrain_name, device)
        self.g_prompt_encoder = self.prompter_encoder
        self.g_prompt_encoder.shared_patch_embedding = None 

    def forward(self, *args, **kwargs):
        return self.g_prompt_encoder(*args, **kwargs)


class GSAMMaskDecoder(GSAMBase):
    def __init__(self, hf_pretrain_name, device="cuda"):
        super().__init__(hf_pretrain_name, device="cuda")
        self.g_mask_decoder = self.mask_decoder

    def forward(self, *args, **kwargs):
        return self.g_mask_decoder(*args, **kwargs)





        
        
