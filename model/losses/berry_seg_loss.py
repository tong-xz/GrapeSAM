import torch
import torch.nn.functional as F

class BerrySegmentationLoss(torch.nn.Module):
    def __init__(self, lambda_overlap=1.0, lambda_size=0.1):
        super().__init__()
        self.lambda_overlap = lambda_overlap
        self.lambda_size = lambda_size

    def forward(self, logits, points):
        """
        logits: 模型输出的logits, shape [B, N, H, W]
        points: berry的点标注, shape [B, N, 2], 每个点是(y, x)坐标
        """
        masks = torch.sigmoid(logits)
        
        # 1. 点一致性损失 (使用logits)
        point_loss = self.point_consistency_loss(logits, points)

        # 2. 重叠惩罚 (使用masks)
        overlap_loss = self.overlap_loss(masks)

        # 3. 大小正则化 (使用masks)
        size_loss = self.size_regularization_loss(masks)

        total_loss = point_loss + self.lambda_overlap * overlap_loss + self.lambda_size * size_loss

        return total_loss

    def point_consistency_loss(self, logits, points):
        batch_size, num_masks, height, width = logits.shape
        
        # 将点坐标转换为mask索引
        point_indices = points[:, :, 1] * width + points[:, :, 0]  # [B, N]
        point_indices = point_indices.unsqueeze(2).expand(-1, -1, height * width)

        # 将logits展平
        flat_logits = logits.view(batch_size, num_masks, -1)  # [B, N, H*W]

        # 获取每个mask在其对应点的logit值
        point_logits = torch.gather(flat_logits, 2, point_indices).squeeze(2)  # [B, N]

        # 计算每个mask在其他点的最大logit值
        other_points_max_logits, _ = (flat_logits.max(dim=2)[0].unsqueeze(1) * (1 - torch.eye(num_masks, device=logits.device))).max(dim=2)  # [B, N]

        # 使用二元交叉熵损失
        loss = F.binary_cross_entropy_with_logits(point_logits, torch.ones_like(point_logits)) + \
               F.binary_cross_entropy_with_logits(other_points_max_logits, torch.zeros_like(other_points_max_logits))

        return loss

    def overlap_loss(self, masks):
        overlaps = torch.sum(masks, dim=1) - torch.max(masks, dim=1)[0]
        return torch.mean(overlaps**2)

    def size_regularization_loss(self, masks):
        mask_sizes = masks.sum(dim=(2, 3))  # [B, N]
        mean_size = mask_sizes.mean()
        return F.mse_loss(mask_sizes, mean_size.expand_as(mask_sizes))