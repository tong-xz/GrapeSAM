import torch
import torch.nn as nn
from torch import Tensor
from transformers import SamConfig
from transformers.models.sam.modeling_sam import (
    SamModel, SamVisionEncoder, SamMaskDecoder, SamPositionalEmbedding, SamPromptEncoder
)
from collections import OrderedDict
import einops
from typing import List, Optional, Dict
import torch.nn.functional as F
import copy

def _build(cfg):
    type = cfg['type']
    if type == 'FeatureAggregator':
        return FeatureAggregator(in_channels=cfg['in_channels'], out_channels=cfg['out_channels'], hidden_channels=cfg["hidden_channels"], select_layers=cfg['select_layers'])
    elif type == 'FeatureSpliter':
        return SimpleFPN(backbone_channel=cfg['backbone_channel'], in_channels=cfg['in_channels'], out_channels=cfg['out_channels'], num_outs=cfg['num_outs'])
    elif type == 'GSAMVisionEncoder':
        ...
    elif type == 'GSAMPromptEncoder':
        return GSAMPromptEncoder(hf_pretrain_name=cfg['hf_pretrain_name'], init_cfg=cfg['init_cfg'])
    elif type == 'GSAMMaskDecoder':
        return GSAMMaskDecoder(hf_pretrain_name=cfg['hf_pretrain_name'], init_cfg=cfg['init_cfg'])
    else:
        return NotImplementedError




class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    

class FPN(nn.Module):
    def __init__(
            self, 
            feature_aggregator=None, 
            feature_spliter=None, 
            init_cfg =None
        ):
        super().__init__()
        if feature_aggregator is not None:
            self.feature_aggregator = FeatureAggregator(feature_aggregator)

        if feature_spliter is not None:
            self.feature_spliter = SimpleFPN(feature_spliter)


    def forward(self, inputs):
        if hasattr(self, 'feature_aggregator'):
            x = self.feature_aggregator(inputs)
        else: 
            x = inputs

        if hasattr(self, 'feature_spliter'):
            x = self.feature_spliter(x)
        else:
            x = (x,)
        return x
    

class FeatureAggregator(nn.Module):
    in_channels_dict = {
        'base': [768] * (12+1),
        'large': [1024] * (24+1),
        'huge': [1280] * (32+1),
    }

    def __init__(
            self,
            in_channels,
            hidden_channels=64,
            out_channels=256,
            select_layers=range(1, 12, 2),
            init_cfg=None,
    ):
        #TODO whether remove init_cfg?
        super().__init__()
        assert isinstance(in_channels, str)
        model_arch = 'base' if 'base' in in_channels else 'large' if 'large' in in_channels else 'huge'
        self.in_channels = self.in_channels_dict[model_arch]
        self.select_layers = select_layers

        self.downconvs = nn.ModuleList()
        for i_layer in self.select_layers:
            self.downconvs.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i_layer], hidden_channels, 1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.hidden_convs = nn.ModuleList()
        for _ in self.select_layers:
            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion_conv = nn.Sequential(
              nn.Conv2d(hidden_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )


    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [einops.rearrange(x, 'b h w c -> b c h w') for x in inputs]

        features = []
        for idx, i_layer in enumerate(self.select_layers):
            features.append(self.downconvs[idx](inputs[i_layer]))

        x = None
        for hidden_state, hidden_conv in zip(features, self.hidden_convs):
            if x is not None:
                 hidden_state = x + hidden_state
            residual = hidden_conv(hidden_state)
            x = hidden_state + residual

        x = self.fusion_conv(x)
        return x


class SimpleFPN(nn.Module):
    def __init__(
            self,
            backbone_channel: int,
            in_channels: List[int],
            out_channels: int,
            num_outs: int,
            conv_cfg: Optional[Dict] = None,
            norm_cfg: Optional[Dict] = None,
            act_cfg: Optional[Dict] = None,
            init_cfg: Optional[Dict] = None
                 ):
        super().__init__()

        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2),
            LN2d( self.backbone_channel // 2, eps=1e-5),
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2,
                               self.backbone_channel // 4, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            #TODO validate structure
            l_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=self.out_channels,
                    kernel_size=1,  # 1x1 卷积
                    stride=1,
                    bias=False
                ),
                LN2d(normalized_shape=256)  # 输出通道是256
            )


            fpn_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                LN2d(normalized_shape=256)  # 使用channel数作为参数
            )
            

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)


    def forward(self, input: Tensor) -> tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        inputs = []
        inputs.append(self.fpn1(input))
        inputs.append(self.fpn2(input))
        inputs.append(self.fpn3(input))
        inputs.append(self.fpn4(input))

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)

class PrompterAnchorRoIPromptHead(nn.Module):
    def __init__(self,):
        super().__init__()


class PrompterAnchorMaskHead(nn.Module):
    def __init__(
            self,
            mask_decoder_cfg,
            in_channels,
            roi_feat_size=14,
            per_pointset_point=5,
            with_sincos=True,
            multimask_output=False,
            attention_similarity=None,
            target_embedding=None,
            output_attentions=None,
            class_agnostic=False,
            loss_mask: dict = dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            init_cfg=None
            ):
        super().__init__()

        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity
        self.target_embedding = target_embedding
        self.output_attentions = output_attentions

        self.mask_decoder = _build(mask_decoder_cfg)

        prompt_encoder_cfg = dict(
            type='GSamPromptEncoder',
            hf_pretrain_name=copy.deepcopy(mask_decoder_cfg.get('hf_pretrain_name')),
            init_cfg=copy.deepcopy(mask_decoder_cfg.get('init_cfg')),
        )
        prompt_encoder = _build(prompt_encoder_cfg)
        self.no_mask_embed = prompt_encoder.prompt_encoder.no_mask_embed

        if with_sincos:
            num_sincos = 2
        else:
            num_sincos = 1
        self.point_emb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_channels*roi_feat_size**2//4, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels * num_sincos * per_pointset_point)
        )
        #TODO cross entropy loss


        self.class_agnostic = class_agnostic

    def forward(self, x, image_embeddings, image_positional_embeddings, roi_img_ids=None):
        img_bs = image_embeddings.shape[0]
        roi_bs = x.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]

        point_embeddings = self.point_emb(x)
        point_embeddings = einops.rearrange(point_embeddings, 'b (n c) -> b n c', n=self.per_pointset_point)
        if self.with_sincos:
            point_embeddings = torch.sin(point_embeddings[..., ::2]) + point_embeddings[..., 1::2]

        # (B * N_set), N_point, C
        sparse_embeddings = point_embeddings.unsqueeze(1)
        num_roi_per_image = torch.bincount(roi_img_ids.long())

        # deal with the case that there is no roi in an image
        num_roi_per_image = torch.cat([num_roi_per_image, torch.zeros(img_bs - len(num_roi_per_image), device=num_roi_per_image.device, dtype=num_roi_per_image.dtype)])

        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(roi_bs, -1, image_embedding_size[0], image_embedding_size[1])
         # get image embeddings with num_roi_per_image
        image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)

        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            attention_similarity=self.attention_similarity,
            target_embedding=self.target_embedding,
            output_attentions=self.output_attentions,
        )
        h, w = low_res_masks.shape[-2:]
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w)
        iou_predictions = iou_predictions.reshape(roi_bs, -1)
        return low_res_masks, iou_predictions

    def get_targets(
            self,
                    ):
                    ...



'''
====================================================================================================
===========================================SAM RELATED==============================================
====================================================================================================
'''
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

        

if __name__ == '__main__':
    config = {'type': 'FeatureAggregator', 'in_channels': 'work_dirs/sam_cache/sam_vit_base', 'out_channels': 256, 'hidden_channels': 32, 'select_layers': range(1, 13, 2)}
    config2 = {'type': 'FeatureSpliter', 'backbone_channel': 256, 'in_channels': [64, 128, 256, 256], 'out_channels': 256, 'num_outs': 5, }
    feature_aggregator, feature_spliter = _build(config), _build(config2)

    inputs = tuple(
        torch.randn(1, 64, 64, 768) 
        for _ in range(13)
    )
    import pdb; pdb.set_trace()
    y = feature_aggregator(inputs)
    y2 = feature_spliter(y)