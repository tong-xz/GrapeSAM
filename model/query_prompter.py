import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import einops
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def denormalize(grid: Tensor) -> Tensor:
    """Denormalize input grid from range [0, 1] to [-1, 1]

    Args:
        grid (torch.Tensor): The grid to be denormalize, range [0, 1].

    Returns:
        torch.Tensor: Denormalized grid, range [-1, 1].
    """

    return grid * 2.0 - 1.0


def point_sample(
    input: Tensor, points: Tensor, align_corners: bool = False, **kwargs
) -> Tensor:
    """A wrapper around :func:`grid_sample` to support 3D point_coords tensors
    Unlike :func:`torch.nn.functional.grid_sample` it assumes point_coords to
    lie inside ``[0, 1] x [0, 1]`` square.

    Args:
        input (torch.Tensor): Feature map, shape (N, C, H, W).
        points (torch.Tensor): Image based absolute point coordinates
            (normalized), range [0, 1] x [0, 1], shape (N, P, 2) or
            (N, Hgrid, Wgrid, 2).
        align_corners (bool, optional): Whether align_corners.
            Default: False

    Returns:
        torch.Tensor: Features of `point` on `input`, shape (N, C, P) or
        (N, C, Hgrid, Wgrid).
    """

    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(2)
    output = F.grid_sample(
        input, denormalize(points), align_corners=align_corners, **kwargs
    )
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertainty(mask_preds: Tensor, labels: Tensor) -> Tensor:
    """Estimate uncertainty based on pred logits.

    We estimate uncertainty as L1 distance between 0.0 and the logits
    prediction in 'mask_preds' for the foreground class in `classes`.

    Args:
        mask_preds (Tensor): mask predication logits, shape (num_rois,
            num_classes, mask_height, mask_width).

        labels (Tensor): Either predicted or ground truth label for
            each predicted mask, of length num_rois.

    Returns:
        scores (Tensor): Uncertainty scores with the most uncertain
            locations having the highest uncertainty score,
            shape (num_rois, 1, mask_height, mask_width)
    """
    if mask_preds.shape[1] == 1:
        gt_class_logits = mask_preds.clone()
    else:
        inds = torch.arange(mask_preds.shape[0], device=mask_preds.device)
        gt_class_logits = mask_preds[inds, labels].unsqueeze(1)
    return -torch.abs(gt_class_logits)


def get_uncertain_point_coords_with_randomness(
    mask_preds: Tensor,
    labels: Tensor,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> Tensor:
    """Get ``num_points`` most uncertain points with random points during
    train.

    Sample points in [0, 1] x [0, 1] coordinate space based on their
    uncertainty. The uncertainties are calculated for each point using
    'get_uncertainty()' function that takes point's logit prediction as
    input.

    Args:
        mask_preds (Tensor): A tensor of shape (num_rois, num_classes,
            mask_height, mask_width) for class-specific or class-agnostic
            prediction.
        labels (Tensor): The ground truth class for each instance.
        num_points (int): The number of points to sample.
        oversample_ratio (float): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled
            via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
            that contains the coordinates sampled points.
    """
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    batch_size = mask_preds.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(
        batch_size, num_sampled, 2, device=mask_preds.device, dtype=mask_preds.dtype
    )
    point_logits = point_sample(mask_preds, point_coords)
    # It is crucial to calculate uncertainty based on the sampled
    # prediction value for the points. Calculating uncertainties of the
    # coarse predictions first and sampling them for points leads to
    # incorrect results.  To illustrate this: assume uncertainty func(
    # logits)=-abs(logits), a sampled point between two coarse
    # predictions with -1 and 1 logits has 0 logits, and therefore 0
    # uncertainty value. However, if we calculate uncertainties for the
    # coarse predictions first, both will have -1 uncertainty,
    # and sampled point will get -1 uncertainty.
    point_uncertainties = get_uncertainty(point_logits, labels)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        batch_size, dtype=torch.long, device=mask_preds.device
    )
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        batch_size, num_uncertain_points, 2
    )
    if num_random_points > 0:
        rand_roi_coords = torch.rand(
            batch_size,
            num_random_points,
            2,
            device=mask_preds.device,
            dtype=mask_preds.dtype,
        )
        point_coords = torch.cat((point_coords, rand_roi_coords), dim=1)
    return point_coords


class RSPrompterQuery(Mask2Former):
    def __init__(self, shared_image_embedding, decoder_freeze=True, *args, **kwargs):
        peft_config = kwargs.get("backbone", {}).get("peft_config", {})
        super().__init__(*args, **kwargs)
        self.decoder_freeze = decoder_freeze
        self.with_mask2formerhead = (
            False if isinstance(self.panoptic_head, RSMask2FormerHead) else True
        )
        self.shared_image_embedding = MODELS.build(
            shared_image_embedding
        )  # There should be a sam image encoder

        self.frozen_modules = []
        if peft_config is None:
            self.frozen_modules += [self.backbone]
        if self.decoder_freeze:
            self.frozen_modules += [
                self.shared_image_embedding,
                self.panoptic_head.mask_decoder,
            ]
        self._set_grad_false(self.frozen_modules)

    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

    def get_image_wide_positional_embeddings(self, size):
        target_device = (
            self.shared_image_embedding.shared_image_embedding.positional_embedding.device
        )
        target_dtype = (
            self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype
        )
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

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        vision_outputs = self.backbone(batch_inputs)
        image_embeddings = vision_outputs[0]
        vision_hidden_states = vision_outputs
        image_positional_embeddings = self.get_image_wide_positional_embeddings(
            size=image_embeddings.shape[-1]
        )
        # repeat with batch size
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(
            batch_size, 1, 1, 1
        )

        x = self.neck(vision_hidden_states)
        return x, image_embeddings, image_positional_embeddings

    def loss(self, batch_inputs: Tensor, batch_data_samples) -> Dict[str, Tensor]:

        x, image_embeddings, image_positional_embeddings = self.extract_feat(
            batch_inputs
        )

        if self.with_mask2formerhead:
            losses = self.panoptic_head.loss(x, batch_data_samples)
        else:
            losses = self.panoptic_head.loss(
                x,
                batch_data_samples,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples, rescale: bool = True):

        x, image_embeddings, image_positional_embeddings = self.extract_feat(
            batch_inputs
        )

        if self.with_mask2formerhead:
            mask_cls_results, mask_pred_results = self.panoptic_head.predict(
                x, batch_data_samples
            )
        else:
            mask_cls_results, mask_pred_results = self.panoptic_head.predict(
                x,
                batch_data_samples,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )

        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results, mask_pred_results, batch_data_samples, rescale=rescale
        )
        results = self.add_pred_to_datasample(batch_data_samples, results_list)

        return results


class SamPromptEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SamPromptEncoder`]. The [`SamPromptEncoder`]
    module is used to encode the input 2D points and bounding boxes. Instantiating a configuration defaults will yield
    a similar configuration to that of the SAM-vit-h
    [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the hidden states.
        image_size (`int`, *optional*, defaults to 1024):
            The expected output resolution of the image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        mask_input_channels (`int`, *optional*, defaults to 16):
            The number of channels to be fed to the `MaskDecoder` module.
        num_point_embeddings (`int`, *optional*, defaults to 4):
            The number of point embeddings to be used.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function in the encoder and pooler.
    """

    def __init__(
        self,
        hidden_size=256,
        image_size=1024,
        patch_size=16,
        mask_input_channels=16,
        num_point_embeddings=4,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_embedding_size = image_size // patch_size
        self.mask_input_channels = mask_input_channels
        self.num_point_embeddings = num_point_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps


class SamPromptEncoder(nn.Module):
    def __init__(self, config: SamPromptEncoderConfig, shared_patch_embedding):
        super().__init__()
        self.shared_embedding = shared_patch_embedding
        self.mask_embed = SamMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)

        self.image_embedding_size = (
            config.image_embedding_size,
            config.image_embedding_size,
        )
        self.input_image_size = config.image_size

        self.point_embed = nn.ModuleList(
            [
                nn.Embedding(1, config.hidden_size) 
                for i in range(config.num_point_embeddings)
            ]
        )
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)

    def _embed_points(
        self, points: torch.Tensor, labels: torch.Tensor, pad: bool
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
            target_labels_shape = (points.shape[0], points.shape[1], 1)
            padding_point = torch.zeros(target_point_shape, device=points.device)
            padding_label = -torch.ones(target_labels_shape, device=labels.device)
            points = torch.cat([points, padding_point], dim=2)
            labels = torch.cat([labels, padding_label], dim=2)
        input_shape = (self.input_image_size, self.input_image_size)
        point_embedding = self.shared_embedding(points, input_shape)

        # torch.where and expanding the labels tensor is required by the ONNX export
        point_embedding = torch.where(
            labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding
        )

        # This is required for the ONNX export. The dtype, device need to be explicitely
        # specificed as otherwise torch.onnx.export interprets as double
        point_embedding = torch.where(
            labels[..., None] != -10,
            point_embedding,
            torch.tensor(
                0.0, dtype=point_embedding.dtype, device=point_embedding.device
            ),
        )

        point_embedding = torch.where(
            (labels == 0)[:, :, :, None],
            point_embedding + self.point_embed[0].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = torch.where(
            (labels == 1)[:, :, :, None],
            point_embedding + self.point_embed[1].weight[None, None, :, :],
            point_embedding,
        )

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        batch_size, nb_boxes = boxes.shape[:2]
        coords = boxes.reshape(batch_size, nb_boxes, 2, 2)
        input_shape = (self.input_image_size, self.input_image_size)
        corner_embedding = self.shared_embedding(coords, input_shape)
        corner_embedding[:, :, 0, :] += self.point_embed[2].weight
        corner_embedding[:, :, 1, :] += self.point_embed[3].weight
        return corner_embedding

    def forward(
        self,
        input_points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        input_labels: Optional[torch.Tensor],
        input_boxes: Optional[torch.Tensor],
        input_masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        sparse_embeddings = None
        batch_size = 1
        target_device = self.shared_embedding.positional_embedding.device
        if input_points is not None:
            batch_size, point_batch_size = input_points.shape[:2]
            if input_labels is None:
                raise ValueError(
                    "If points are provided, labels must also be provided."
                )
            point_embeddings = self._embed_points(
                input_points, input_labels, pad=(input_boxes is None)
            )
            sparse_embeddings = point_embeddings
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat(
                    [sparse_embeddings, box_embeddings], dim=2
                )
        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size,
                -1,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
            )

        if sparse_embeddings is None:
            sparse_embeddings = torch.zeros(
                (batch_size, 1, 1, self.hidden_size), device=target_device
            )

        return sparse_embeddings, dense_embeddings


class RSSamPromptEncoder(SamPromptEncoder, BaseModule):
    def __init__(
        self,
        hf_pretrain_name,
        extra_config=None,
        init_cfg=None,
    ):
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).prompt_encoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.prompt_encoder = SamPromptEncoder(sam_config, shared_patch_embedding=None)

    def forward(self, *args, **kwargs):
        return self.prompt_encoder(*args, **kwargs)


class RSMask2FormerHead(Mask2FormerHead):
    def __init__(
        self,
        mask_decoder,
        decoder_plus,
        with_sincos=True,
        per_pointset_point=1,
        multimask_output=False,
        attention_similarity=None,
        target_embedding=None,
        output_attentions=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.decoder_plus = decoder_plus
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity
        self.target_embedding = target_embedding
        self.output_attentions = output_attentions

        self.mask_decoder = MODELS.build(mask_decoder)

        prompt_encoder = dict(
            type="RSSamPromptEncoder",
            hf_pretrain_name=copy.deepcopy(mask_decoder.get("hf_pretrain_name")),
            init_cfg=copy.deepcopy(mask_decoder.get("init_cfg")),
        )
        prompt_encoder = MODELS.build(prompt_encoder)
        prompt_encoder.init_weights()
        if self.decoder_plus:
            self.sam_mask_embed = prompt_encoder.prompt_encoder.mask_embed
        else:
            self.no_mask_embed = prompt_encoder.prompt_encoder.no_mask_embed
            del self.mask_embed

        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos

        self.feat_channels = kwargs["feat_channels"]
        self.out_channels = kwargs["out_channels"]
        if with_sincos:
            num_sincos = 2
        else:
            num_sincos = 1
        self.point_emb = nn.Sequential(
            nn.Linear(self.feat_channels, self.feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels // 2, self.feat_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(
                self.feat_channels // 2,
                self.out_channels * num_sincos * per_pointset_point,
            ),
        )
        del self.cls_embed
        self.cls_embed = nn.Sequential(
            nn.Linear(self.feat_channels, self.feat_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels, self.num_classes + 1),
        )

    def _forward_head(
        self,
        decoder_out: Tensor,
        mask_feature: Tensor,
        attn_mask_target_size: Tuple[int, int],
        image_embeddings=None,
        image_positional_embeddings=None,
    ) -> Tuple[Tensor]:
        img_bs = image_embeddings.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]

        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (batch_size, num_queries, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (batch_size, num_queries, c)
        point_embedings = self.point_emb(decoder_out)

        point_embedings = einops.rearrange(
            point_embedings,
            "b n_set (n_point c) -> b n_set n_point c",
            n_point=self.per_pointset_point,
        )
        if self.with_sincos:
            point_embedings = (
                torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2]
            )

        # B, N_set, N_point, C => (B, N_set), 1, N_point, C
        sparse_embeddings = einops.rearrange(
            point_embedings, "b n_set n_point c -> (b n_set) n_point c"
        )
        sparse_embeddings = sparse_embeddings.unsqueeze(1)

        if self.decoder_plus:
            # shape (num_queries, batch_size, h, w)
            mask_embed = self.mask_embed(decoder_out)
            mask_pred_plus = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)

            input_masks = mask_pred_plus.detach()
            input_masks = einops.repeat(input_masks, "b n h w -> (b n) c h w", c=1)
            # (bs num_q) c h w
            dense_embeddings = self.sam_mask_embed(input_masks)
        else:
            mask_pred_plus = None
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                img_bs, -1, image_embedding_size[0], image_embedding_size[1]
            )

        image_embeddings = torch.repeat_interleave(
            image_embeddings, repeats=self.num_queries, dim=0
        )
        image_positional_embeddings = torch.repeat_interleave(
            image_positional_embeddings, repeats=self.num_queries, dim=0
        )
        mask_pred, iou_predictions, mask_dencoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            attention_similarity=self.attention_similarity,
            target_embedding=self.target_embedding,
            output_attentions=self.output_attentions,
        )
        mask_pred = mask_pred.reshape(img_bs, -1, *mask_pred.shape[-2:])
        if not self.decoder_plus:
            h, w = mask_pred.shape[-2:]
            # shape (batch_size, num_queries, h, w)
            attn_mask_pred = mask_pred.reshape(img_bs, -1, h, w)
        else:
            attn_mask_pred = mask_pred_plus
        attn_mask = F.interpolate(
            attn_mask_pred, attn_mask_target_size, mode="bilinear", align_corners=False
        )
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = (
            attn_mask.flatten(2)
            .unsqueeze(1)
            .repeat((1, self.num_heads, 1, 1))
            .flatten(0, 1)
        )
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()
        return cls_pred, mask_pred, attn_mask, mask_pred_plus

    def forward(
        self,
        x: List[Tensor],
        batch_data_samples,
        image_embeddings=None,
        image_positional_embeddings=None,
    ) -> Tuple[List[Tensor]]:
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask).to(
                decoder_input.dtype
            )
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        mask_pred_plus_list = []
        attn_mask = None

        cls_pred, mask_pred, attn_mask, mask_pred_plus = self._forward_head(
            query_feat,
            mask_features,
            multi_scale_memorys[0].shape[-2:],
            image_embeddings,
            image_positional_embeddings,
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        mask_pred_plus_list.append(mask_pred_plus)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            if attn_mask is not None:
                # if a mask is all True(all background), then set it all False.
                mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
                attn_mask = attn_mask & mask_sum

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask, mask_pred_plus = self._forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                    -2:
                ],
                image_embeddings,
                image_positional_embeddings,
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            mask_pred_plus_list.append(mask_pred_plus)
        return cls_pred_list, mask_pred_list, mask_pred_plus_list

    def loss(
        self,
        x: Tuple[Tensor],
        batch_data_samples,
        image_embeddings=None,
        image_positional_embeddings=None,
    ) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation of the panoptic
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if "gt_sem_seg" in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
            else:
                batch_gt_semantic_segs.append(None)

        # forward
        all_cls_scores, all_mask_preds, all_mask_preds_plus = self(
            x, batch_data_samples, image_embeddings, image_positional_embeddings
        )
        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(
            batch_gt_instances, batch_gt_semantic_segs
        )
        # loss
        losses = self.loss_by_feat(
            all_cls_scores,
            all_mask_preds,
            all_mask_preds_plus,
            batch_gt_instances,
            batch_img_metas,
        )
        return losses

    def loss_by_feat(
        self,
        all_cls_scores: Tensor,
        all_mask_preds: Tensor,
        all_mask_preds_plus,
        batch_gt_instances: List,
        batch_img_metas: List[dict],
    ) -> Dict[str, Tensor]:
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [batch_gt_instances for _ in range(num_dec_layers)]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice, losses_mask_plus, losses_dice_plus = (
            multi_apply(
                self._loss_by_feat_single,
                all_cls_scores,
                all_mask_preds,
                all_mask_preds_plus,
                batch_gt_instances_list,
                img_metas_list,
            )
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_mask"] = losses_mask[-1]
        loss_dict["loss_dice"] = losses_dice[-1]
        loss_dict["loss_mask_plus"] = losses_mask_plus[-1]
        loss_dict["loss_dice_plus"] = losses_dice_plus[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for (
            loss_cls_i,
            loss_mask_i,
            loss_dice_i,
            loss_mask_plus_i,
            loss_dice_plus_i,
        ) in zip(
            losses_cls[:-1],
            losses_mask[:-1],
            losses_dice[:-1],
            losses_mask_plus[:-1],
            losses_dice_plus[:-1],
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_mask"] = loss_mask_i
            loss_dict[f"d{num_dec_layer}.loss_dice"] = loss_dice_i
            loss_dict[f"d{num_dec_layer}.loss_mask_plus"] = loss_mask_plus_i
            loss_dict[f"d{num_dec_layer}.loss_dice_plus"] = loss_dice_plus_i

            num_dec_layer += 1
        return loss_dict

    def _loss_by_feat_single(
        self,
        cls_scores: Tensor,
        mask_preds: Tensor,
        mask_preds_plus,
        batch_gt_instances: List,
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor]:
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        mask_preds_plus_list = [mask_preds_plus[i] for i in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            mask_targets_list,
            mask_weights_list,
            avg_factor,
        ) = self.get_targets(
            cls_scores_list, mask_preds_plus_list, batch_gt_instances, batch_img_metas
        )

        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=class_weight[labels].sum()
        )

        num_total_masks = cls_scores.new_tensor([avg_factor]).mean()
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]
        mask_preds_plus = mask_preds_plus[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            loss_dice_plus = mask_preds_plus.sum()
            loss_mask_plus = mask_preds_plus.sum()
            return loss_cls, loss_mask, loss_dice, loss_mask_plus, loss_dice_plus

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1),
                None,
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # points_coords = points_coords.to(mask_preds.dtype)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).to(mask_preds.dtype), points_coords
            ).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(mask_preds.unsqueeze(1), points_coords).squeeze(
            1
        )
        mask_point_preds_plus = point_sample(
            mask_preds_plus.unsqueeze(1), points_coords
        ).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks
        )
        loss_dice_plus = self.loss_dice(
            mask_point_preds_plus, mask_point_targets, avg_factor=num_total_masks
        )

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)

        mask_point_preds_plus = mask_point_preds_plus.reshape(-1)

        # loss_mask = self.loss_mask(
        #     mask_point_preds,
        #     mask_point_targets,
        #     avg_factor=num_total_masks * self.num_points)
        # to avoid nan in fp16 when num_total_masks * self.num_points
        loss_mask = self.loss_mask(mask_point_preds, mask_point_targets)
        loss_mask_plus = self.loss_mask(mask_point_preds_plus, mask_point_targets)
        return loss_cls, loss_mask, loss_dice, loss_mask_plus, loss_dice_plus

    def predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples,
        image_embeddings=None,
        image_positional_embeddings=None,
    ) -> Tuple[Tensor]:
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        all_cls_scores, all_mask_preds, all_mask_preds_plus = self(
            x,
            batch_data_samples,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        mask_pred_plus_results = all_mask_preds_plus[-1]
        # upsample masks
        try:
            img_shape = batch_img_metas[0]["batch_input_shape"]
        except:
            img_shape = batch_img_metas[0]["pad_shape"]
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode="bilinear",
            align_corners=False,
        )

        return mask_cls_results, mask_pred_results
