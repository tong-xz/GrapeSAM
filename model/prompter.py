import torch
import torch.nn as nn
from torch import Tensor


import einops
from typing import List, Optional, Dict, Tuple
import torch.nn.functional as F
import copy

def build(cfg):
    type = cfg['type']
    if type == 'FeatureAggregator':
        return FeatureAggregator(in_channels=cfg['in_channels'], out_channels=cfg['out_channels'], hidden_channels=cfg["hidden_channels"], select_layers=cfg['select_layers'])
    elif type == 'FeatureSpliter':
        return SimpleFPN(backbone_channel=cfg['backbone_channel'], in_channels=cfg['in_channels'], out_channels=cfg['out_channels'], num_outs=cfg['num_outs'])
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

from .utils import SinePositionalEncoding
# TODO how to convert base roi functions
class PrompterAnchorRoIPromptHead(nn.Module):
    def __init__(self, with_extra_pe):
        super().__init__()
        if with_extra_pe:
            #TODO bbox_roi_extractor
            out_channels = self.bbox_roi_extractor.out_channels
            positional_encoding = dict(num_feats = out_channels // 2, normalize = True)
            self.extra_pe = SinePositionalEncoding(**positional_encoding)

        self.mask_roi_extractor = None
        self.mask_head = None
        self.bbox_assigner = None
        self.bbox_sampler = None
        self.bbox_loss = None
        self.mask_loss = None



    def _mask_forward(
            self,
            x: Tuple[Tensor],
            rois: Tensor=None,
            pos_inds: Optional[Tensor] = None,
            bbox_feats: Optional[Tensor] =None,
            image_embeddings = None,
            image_positional_embeddings = None
    ) -> dict:
        
        assert ((rois is not None) ^ (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
            #TODO with shared head?
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds, iou_predictions = self.mask_head(
            mask_feats,
            image_embeddings = image_embeddings,
            image_positional_embeddings = image_positional_embeddings,
            roi_img_ids=rois[:, 0] if rois is not None else None,
        )
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats, iou_predictions=iou_predictions)
        return mask_results
    
    '''
    replace SamplingResult type to list[dict]
    replace InstanceData type to dict
    
    '''
    def mask_loss(
            self,
            x: Tuple[Tensor],
            sampling_results: List[dict],
            bbox_feats: Tensor,
            batch_gt_instances: List[dict],
            image_embeddings=None,
            image_positional_embeddings=None
    ) -> dict:
        
        #TODO self.share_roi_extractor where comes from?
        if not self.share_roi_extractor:
            from model.utils import bbox2roi
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            if len(pos_rois) == 0:
                print('no pos rois')
                return dict(loss_mask=dict(0 * x[0].sum()))
            
            mask_results = self._mask_forward(x, pos_rois, 
                                              image_embeddings=image_embeddings, 
                                              image_positional_embeddings=image_positional_embeddings)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))            
            pos_inds = torch.cat(pos_inds)
            
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)
            
        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds = mask_results['mask_preds'],
            sampling_results = sampling_results,
            batch_gt_instances = batch_gt_instances,
            rcnn_train_cfg = self.train_cfg 
        )
        mask_results.update(loss_mask = mask_loss_and_target['loss_mask'])
        return mask_results


    def loss(
            self,
            x: Tuple[Tensor],
            rpn_results_list: List[Dict],
            batch_data_samples: List[Dict],
            # extra inputs
            image_embeddings = None,
            image_positional_embeddings = None

    ) -> dict:
        assert len(rpn_results_list) == len(batch_data_samples)
        from .utils import unpack_gt_instances

        batch_gt_instances, batch_gt_instances_ignore, _ = unpack_gt_instances(batch_data_samples)

        if hasattr(self, 'extra_pe'):
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
                outputs.append(output)
            x = tuple(outputs)
            
        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []

        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            #TODO what are these two modules?

            assign_result = self.bbox_assigner.assign(
                rpn_results,
                batch_gt_instances[i],
                batch_gt_instances_ignore[i]
            )
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats = [lvl_feat[i][None] for lvl_feat in x]
            )
            sampling_results.append(sampling_result)

        losses = dict()
        
        #TODO BBOX HEAD LOSS how to solve?
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        #TODO MASK HEAD LOSS how to solve?
        if self.with_mask:
            mask_results = self.mask_loss(
                x, sampling_result, bbox_results['bbox_feats'], batch_gt_instances,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings
            )
            losses.update(mask_results['loss_mask'])


        return losses

    # TODO two prediction functions for inference
    def predict_mask(
            self,
    ) -> List:
        ...

    def predict(
            
    ) -> List:















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

        self.mask_decoder = build(mask_decoder_cfg)

        prompt_encoder_cfg = dict(
            type='GSamPromptEncoder',
            hf_pretrain_name=copy.deepcopy(mask_decoder_cfg.get('hf_pretrain_name')),
            init_cfg=copy.deepcopy(mask_decoder_cfg.get('init_cfg')),
        )
        prompt_encoder = build(prompt_encoder_cfg)
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
        self.loss_mask = []
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
            sampling_results: List[dict],
            batch_gt_instances: List[dict],
            rcnn_train_cfg: Dict) -> Tensor:
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]

        gt_masks = [res.masks for res in batch_gt_instances]
        mask_targets_list = []
        mask_size = rcnn_train_cfg.mask_size
        device = pos_proposals[0].device

        for pos_gt_inds, gt_mask in zip(pos_assigned_gt_inds, gt_masks):
            if len(pos_gt_inds)==0:
                mask_targets = torch.zeros((0,) + mask_size, device=device, dtype=torch.float32)
            else:
                mask_targets = gt_mask[pos_gt_inds.cpu()].to_tensor(dtype=torch.float32, device=device)
            mask_targets_list.append(mask_targets)
        mask_targets = torch.cat(mask_targets_list)
        return mask_targets
    

    def loss_and_target(
            self,
            mask_preds: Tensor,
            sampling_results: List[dict],
            batch_gt_instances: List[dict],
            rcnn_train_cfg: Dict
    ) -> dict:
        mask_targets = self.get_targets(sampling_results=sampling_results, batch_gt_instances=batch_gt_instances, rcnn_train_cfg=rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        mask_preds = F.interpolate(mask_preds, size=mask_targets.shape[-2:], mode='bilinear', align_corners=False)
        loss = Dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets, torch.zeros_like(pos_labels))
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets, pos_labels)
        loss['loss_mask'] = loss_mask
        return dict(loss_mask=loss, mask_targets=mask_targets)


 
        

if __name__ == '__main__':
    config = {'type': 'FeatureAggregator', 'in_channels': 'work_dirs/sam_cache/sam_vit_base', 'out_channels': 256, 'hidden_channels': 32, 'select_layers': range(1, 13, 2)}
    config2 = {'type': 'FeatureSpliter', 'backbone_channel': 256, 'in_channels': [64, 128, 256, 256], 'out_channels': 256, 'num_outs': 5, }
    feature_aggregator, feature_spliter = build(config), build(config2)

    inputs = tuple(
        torch.randn(1, 64, 64, 768) 
        for _ in range(13)
    )
    import pdb; pdb.set_trace()
    y = feature_aggregator(inputs)
    y2 = feature_spliter(y)