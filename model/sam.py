import torch
import torch.nn as nn
from transformers import SamConfig
from transformers.models.sam.modeling_sam import (
    SamModel, SamVisionEncoder, SamMaskDecoder, SamPositionalEmbedding, SamPromptEncoder
)
from collections import OrderedDict
from typing import Optional, Dict


'''
====================================================================================================
===========================================SAM RELATED==============================================
====================================================================================================
'''


def build_gsam(cfg: Dict):
    type = cfg['type']
    if type == 'GSAMVisionEncoder':
        return GSAMVisionEncoder(hf_pretrain_name=cfg['hf_pretrain_name'], init_cfg=cfg['init_cfg'], extra_cfg=cfg['extra_cfg'], device=cfg['device'])
    elif type == 'GSAMPromptEncoder':
        return GSAMPromptEncoder(hf_pretrain_name=cfg['hf_pretrain_name'], init_cfg=cfg['init_cfg'])
    elif type == 'GSAMMaskDecoder':
        return GSAMMaskDecoder(hf_pretrain_name=cfg['hf_pretrain_name'], init_cfg=cfg['init_cfg'], extra_cfg=cfg['extra_cfg'], device=cfg['device'])
    else:
        return NotImplementedError


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
        
        # TODO  'shared_image_embedding.positional_embedding'   'prompt_encoder.shared_embedding.positional_embedding', position

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        total_params = sum(param.numel() for param in new_state_dict.values())
        total_size = sum(param.element_size() * param.numel() for param in new_state_dict.values())
        print(f"{name} weight successfully loaded {total_size/1024/1024:.2f} MB")
    
        if len(missing_keys) > 0:
            print(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys}")

class GSAMVisionEncoder(nn.Module):
    '''
        hf_pretrain_name: 'work_dirs/sam_cache/sam_vit_base'
        extra_cfg: {'output_hidden_states': True}
        peft_config: none
        init_cfg: {'type': 'Pretrained', 'checkpoint': 'work_dirs/sam_cache/sam_vit_base/pytorch_model.bin'}
    
    '''
    def __init__(self, hf_pretrain_name, init_cfg, extra_cfg, device):
        super().__init__()
        # load config
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_cfg is not None:
            sam_config.update(extra_cfg)

        self.vision_encoder = SamVisionEncoder(sam_config).to(device)
        if init_cfg is not None:
            _load_weights(self.vision_encoder, 'vision_encoder.', init_cfg['checkpoint'], device)
    
    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)


class GSAMPromptEncoder(nn.Module):
    def __init__(self, hf_pretrain_name, init_cfg, extra_cfg, device):
        super().__init__()
        # load config
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).prompt_encoder_config
        if extra_cfg is not None:
            sam_config.update(extra_cfg)
    
        self.prompt_encoder = SamPromptEncoder(sam_config, shared_patch_embedding=None)
        if init_cfg is not None:
            _load_weights(self.prompt_encoder, 'prompt_encoder.', init_cfg['checkpoint'], device)

    def forward(self, *args, **kwargs):
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (`torch.Tensor`, *optional*):
                point coordinates and labels to embed.
            boxes (`torch.Tensor`, *optional*):
                boxes to embed
            masks (`torch.Tensor`, *optional*):
                masks to embed
        """
        return self.prompt_encoder(*args, **kwargs)


class GSAMMaskDecoder(nn.Module):
    def __init__(self, hf_pretrain_name, init_cfg, extra_cfg, device):
        super().__init__()
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).mask_decoder_config
        
        if extra_cfg is not None:
            sam_config.update(extra_cfg)
        

        self.mask_decoder = SamMaskDecoder(sam_config)
        
        if init_cfg is not None:
            _load_weights(self.mask_decoder, 'mask_decoder.', init_cfg['checkpoint'], device)

    def forward(self, *args, **kwargs):
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`torch.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`torch.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        
        return self.mask_decoder(*args, **kwargs)



class GSAMPositionalEmbedding(nn.Module):
    def __init__(self, hf_pretrain_name, init_cfg, extra_cfg, device):
        super().__init__()

        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_cfg is not None:
            sam_config.update(extra_cfg)

        self.shared_image_embedding = SamPositionalEmbedding(sam_config)

        if init_cfg is not None:
            _load_weights(self.shared_image_embedding, 'shared_image_embedding.', init_cfg['checkpoint'], device)

    def forward(self, *args, **kwargs):
        return self.shared_image_embedding(*args, **kwargs)
