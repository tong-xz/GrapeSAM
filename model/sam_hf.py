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
    SamVisionLayer,
    SamVisionConfig,
    SamVisionEncoderOutput,
)
from collections import OrderedDict
from typing import Optional, Dict, Tuple, Union
from transformers.modeling_utils import load_state_dict
from .adapter import ViTAdapters


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

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = ViTLayer(
                config,
                window_size=(
                    config.window_size if i not in config.global_attn_indexes else 0
                ),
            )
            self.layers.append(layer)

    def forward(self, *args, **kwargs):
        # Call parent class forward method
        return super().forward(*args, **kwargs)


class GSamVisionEncoderFT(SamVisionEncoder):
    def __init__(self, config: SamVisionConfig):
        super().__init__(config)

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = ViTLayer(
                config,
                window_size=(
                    config.window_size if i not in config.global_attn_indexes else 0
                ),
            )
            self.layers.append(layer)

    @torch.no_grad()
    def patch_embed_no_grad(self, x):
        return self.patch_embed(x)

    @torch.enable_grad()
    def patch_embed_grad(self, x):
        return self.patch_embed(x)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        patch_embed_grad: Optional[bool] = False,
    ) -> Union[Tuple, SamVisionEncoderOutput]:

        adapter = ViTAdapters(
            adapter_layer=range(8, 33, 2),
            embed_dim=1280,
            use_color_adapter=True,
            use_space_adapter=True,
            use_mlp_adapter=True,
        ).cuda()

        # 设置输出参数（与父类相同）
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 根据 patch_embed_grad 参数选择使用哪个 patch embedding 方法
        if patch_embed_grad:
            hidden_states = self.patch_embed_grad(pixel_values)
        else:
            hidden_states = self.patch_embed_no_grad(pixel_values)

        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历层并应用适配器
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                # 添加适配器支持
                layer_outputs = layer_module(
                    hidden_states,
                    output_attentions=output_attentions,
                    adapter=getattr(adapter, f"adapter_{i}", None),
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.neck(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ViTLayer(SamVisionLayer):
    def __init__(self, config, window_size):
        super().__init__(config, window_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        adapter: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.FloatTensor]:
        # 调用父类的 forward 方法来处理基本逻辑
        residual = hidden_states

        # 使用父类的 layer_norm1，但包装在 no_grad 中
        with torch.no_grad():
            hidden_states = self.layer_norm1(hidden_states)

        # Window partition
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(
                hidden_states, self.window_size
            )

        # 使用父类的 attn，但包装在 no_grad 中
        with torch.no_grad():
            hidden_states, attn_weights = self.attn(
                hidden_states=hidden_states,
                output_attentions=output_attentions,
            )

        # 添加空间适配器
        if getattr(adapter, "space_adapter", False):
            hidden_states = adapter.space_adapter(hidden_states)

        # Reverse window partition
        if self.window_size > 0:
            hidden_states = self.window_unpartition(
                hidden_states, self.window_size, padding_shape, (height, width)
            )

        # 添加颜色适配器
        if getattr(adapter, "color_adapter", False):
            hidden_states = hidden_states * adapter.color_adapter(hidden_states)
        hidden_states = residual + hidden_states

        # 使用父类的 layer_norm2，但包装在 no_grad 中
        with torch.no_grad():
            layernorm_output = self.layer_norm2(hidden_states)

        # 添加 MLP 适配器
        if getattr(adapter, "mlp_adapter", False):
            hidden_states = (
                hidden_states
                + self.mlp(layernorm_output)
                + adapter.mlp_adapter(hidden_states)
            )
        else:
            hidden_states = hidden_states + self.mlp(layernorm_output)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


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
