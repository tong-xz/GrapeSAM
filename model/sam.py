import torch
import torch.nn as nn
from transformers import SamConfig
from transformers.models.sam.modeling_sam import (
    SamModel, SamVisionEncoder, SamMaskDecoder, SamPositionalEmbedding, SamPromptEncoder
)
from collections import OrderedDict


def _load_weights(model, name, ckpt_path, device):
        '''
        model: self.vision_encoder
        name: 'vision_encoder.'
        '''
        state_dict = torch.load(ckpt_path, map_location=device)
        new_state_dict = OrderedDict()
        
        for key, value in state_dict.items():
            if key.startswith(name):
                new_key = key.replace(name, '')
                new_state_dict[new_key] = value
        
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        total_params = sum(param.numel() for param in new_state_dict.values())
        total_size = sum(param.element_size() * param.numel() for param in new_state_dict.values())
        print(f"{name} weight successfully loaded, state dict size: {len(new_state_dict)} layers, {total_params:,} parameters, {total_size/1024/1024:.2f} MB")
    
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")


class GSAMVisionEncoder(nn.Module):
    def __init__(self, hf_pretrain_name, init_cfg, extra_config, device):
        super().__init__()
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        
        if extra_config is not None:
            sam_config.update(extra_config)

        self.vision_encoder = SamVisionEncoder(sam_config).to(device)
        if init_cfg is not None:
            _load_weights(self.vision_encoder, 'vision_encoder.', init_cfg['checkpoint'], device)
    
    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)


# class GSAMPromptEncoder(GSAMBase):
#     def __init__(self, hf_pretrain_name, device):
#         super().__init__(hf_pretrain_name, device)
#         self.g_prompt_encoder = self.prompter_encoder
#         self.g_prompt_encoder.shared_patch_embedding = None 

#     def forward(self, *args, **kwargs):
#         return self.g_prompt_encoder(*args, **kwargs)


# class GSAMMaskDecoder(GSAMBase):
#     def __init__(self, hf_pretrain_name, device="cuda"):
#         super().__init__(hf_pretrain_name, device="cuda")
#         self.g_mask_decoder = self.mask_decoder

#     def forward(self, *args, **kwargs):
#         return self.g_mask_decoder(*args, **kwargs)





        
        
