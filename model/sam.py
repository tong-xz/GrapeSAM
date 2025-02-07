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
    SamImageSegmentationOutput,
)
from collections import OrderedDict
from typing import List, Optional, Dict, Tuple, Union
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

    @torch.no_grad()
    def get_image_embeddings(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns the image embeddings by passing the pixel values through the vision encoder.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        vision_output = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeddings = vision_output[0]
        return image_embeddings

    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
    ):
        r"""
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`torch.LongTensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return prompt_output

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("facebook/sam-vit-base")
        >>> processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

        >>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        >>> input_points = [[[400, 650]]]  # 2D location of a window on the car
        >>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

        >>> # Get segmentation mask
        >>> outputs = model(**inputs)

        >>> # Postprocess masks
        >>> masks = processor.post_process_masks(
        ...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        ... )
        ```
        """
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

        if pixel_values is None and image_embeddings is None:
            raise ValueError(
                "Either pixel_values or image_embeddings must be provided."
            )

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError(
                "Only one of pixel_values and image_embeddings can be provided."
            )

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
                " got {}.".format(input_points.shape),
            )
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
                " got {}.".format(input_boxes.shape),
            )
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                        point_batch_size, box_batch_size
                    )
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = (
            pixel_values.shape[0]
            if pixel_values is not None
            else image_embeddings.shape[0]
        )
        image_positional_embeddings = image_positional_embeddings.repeat(
            batch_size, 1, 1, 1
        )

        vision_attentions = None
        vision_hidden_states = None

        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeddings = vision_outputs[0]

            if output_hidden_states:
                vision_hidden_states = vision_outputs[1]
            if output_attentions:
                vision_attentions = vision_outputs[-1]

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(
                input_points[:, :, :, 0], dtype=torch.int, device=input_points.device
            )

        if (
            input_points is not None
            and image_embeddings.shape[0] != input_points.shape[0]
        ):
            raise ValueError(
                "The batch size of the image embeddings and the input points must be the same. ",
                "Got {} and {} respectively.".format(
                    image_embeddings.shape[0], input_points.shape[0]
                ),
                " if you want to pass multiple points for the same image, make sure that you passed ",
                " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
                " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
            )

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )

        if not return_dict:
            output = (iou_predictions, low_res_masks)
            if output_hidden_states:
                output = output + (vision_hidden_states,)

            if output_attentions:
                output = output + (vision_attentions, mask_decoder_attentions)
            return output

        return SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
        )


class GSamVisionEncoder(SamVisionEncoder):
    def __init__(self, config):
        # Call parent class constructor
        super().__init__(config)

        # self.layers = nn.ModuleList()
        # for i in range(config.num_hidden_layers):
        #     layer = ViTLayer(
        #         config,
        #         window_size=(
        #             config.window_size if i not in config.global_attn_indexes else 0
        #         ),
        #     )
        #     self.layers.append(layer)

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


def predict_by_points(
    model, processor, raw_image, points, multimask_output, optimal=True, device="cuda"
):
    inputs = processor(raw_image, input_points=points, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=multimask_output)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = outputs.iou_scores

    if optimal:
        # Get the best mask for each prediction
        scores = scores.squeeze(0)  # Shape: [N, 3]
        best_mask_indices = torch.argmax(scores, dim=1)  # Shape: [N]

        # Select the best masks using the indices
        masks = masks[0]  # Shape: [N, 3, H, W]
        N, _, H, W = masks.shape
        best_masks = torch.zeros((N, 1, H, W), device=masks.device)
        for i in range(N):
            best_masks[i, 0] = masks[i, best_mask_indices[i]]

        # Get corresponding best scores
        best_scores = torch.gather(scores, 1, best_mask_indices.unsqueeze(1)).squeeze(
            1
        )  # Shape: [N]

        return best_masks, best_scores
    else:
        return masks, scores
