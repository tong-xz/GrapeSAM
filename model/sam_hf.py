import torch
import torch.nn as nn
from transformers import SamConfig, PreTrainedModel
from transformers.models.sam.modeling_sam import (
    SamModel,
    SamPreTrainedModel,
    SamVisionEncoder,
    SamMaskDecoder,
    SamPositionalEmbedding,
    SamPromptEncoder,
)
from collections import OrderedDict
from typing import Optional, Dict
from transformers.modeling_utils import load_state_dict


"""
====================================================================================================
===========================================SAM RELATED==============================================
====================================================================================================
"""


class GSamModel(SamPreTrainedModel):
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, config):
        super().__init__(config)
        self.shared_image_embedding = SamPositionalEmbedding(config.vision_config)

        self.vision_encoder = GSamVisionEncoder(config.vision_config)
        self.prompt_encoder = GSamPromptEncoder(
            config.prompt_encoder_config,
            self.shared_image_embedding,
        )
        self.mask_decoder = GSamMaskDecoder(config.mask_decoder_config)

        self.post_init()

    def get_input_embeddings(self):
        return SamModel.get_input_embeddings(self)

    def get_image_wide_positional_embeddings(self):
        size = self.config.prompt_encoder_config.image_embedding_size
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(
            torch.stack([x_embed, y_embed], dim=-1)
        )
        return positional_embedding.permute(2, 0, 1).unsqueeze(
            0
        )  # channel x height x width


class GSamVisionEncoder(SamVisionEncoder):
    def __init__(self, config):
        # Call parent class constructor
        super().__init__(config)

    def forward(self, *args, **kwargs):
        # Call parent class forward method
        return super().forward(*args, **kwargs)


class GSamPromptEncoder(SamPromptEncoder):
    def __init__(self, prompt_encoder_config, shared_image_embedding):
        super().__init__(
            prompt_encoder_config, shared_patch_embedding=shared_image_embedding
        )

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class GSamMaskDecoder(SamMaskDecoder):
    def __init__(self, mask_decoder_config):
        super().__init__(mask_decoder_config)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class GSamPositionalEmbedding(SamMaskDecoder):
    def __init__(self, vision_config):
        super().__init__(vision_config)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
