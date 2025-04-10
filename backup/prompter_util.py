
# ----------------Prompter related--------------------------------
# Auxiliary functions
class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions.

    This implementation follows the method described in
    'End-to-End Object Detection with Transformers' (https://arxiv.org/pdf/2005.12872).

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. Used only when `normalize` is True. Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): Offset added to embedding when doing normalization.
            Defaults to 0.
    """

    def __init__(
        self,
        num_feats: int,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
    ) -> None:
        super().__init__()
        if normalize:
            assert isinstance(
                scale, (float, int)
            ), "When normalize is set, scale should be a float or int."
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(
        self, mask: torch.Tensor, input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function for SinePositionalEncoding.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values represent
                ignored positions, while zero values mean valid positions.
                Shape [bs, h, w].
            input (Tensor, optional): Input image/feature Tensor.
                Shape [bs, c, h, w].

        Returns:
            pos (Tensor): Position embedding with shape [bs, num_feats*2, h, w].
        """
        assert not (
            mask is None and input is None
        ), "Either 'mask' or 'input' must be provided."

        if mask is not None:
            B, H, W = mask.size()
            device = mask.device
            mask = mask.to(torch.int)
            not_mask = 1 - mask  # Logical NOT operation
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            B, _, H, W = input.shape
            device = input.device
            x_embed = (
                torch.arange(1, W + 1, dtype=torch.float32, device=device)
                .view(1, 1, -1)
                .repeat(B, H, 1)
            )
            y_embed = (
                torch.arange(1, H + 1, dtype=torch.float32, device=device)
                .view(1, -1, 1)
                .repeat(B, 1, W)
            )

        if self.normalize:
            y_embed = (
                (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            )
            x_embed = (
                (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
            )

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self) -> str:
        """String representation of the module."""
        return (
            f"{self.__class__.__name__}(num_feats={self.num_feats}, "
            f"temperature={self.temperature}, normalize={self.normalize}, "
            f"scale={self.scale}, eps={self.eps})"
        )


def CrossEntropyLoss(): ...


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


def bbox2roi(bbox_list: List[Tensor]) -> Tensor:
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (List[Tensor]): A list of bbox tensors corresponding to a batch
            of images. Each tensor has shape (n, 4) where n is the number of boxes
            and the 4 columns represent [x1, y1, x2, y2].

    Returns:
        Tensor: shape (n, 5) where n is the total number of boxes across all images.
            Each row contains [batch_ind, x1, y1, x2, y2] where batch_ind indicates
            which image the box belongs to.
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        # Ensure bboxes is a tensor
        if not isinstance(bboxes, Tensor):
            bboxes = torch.tensor(bboxes, dtype=torch.float32)

        # Create image index column
        img_inds = torch.full(
            (bboxes.size(0), 1), img_id, dtype=bboxes.dtype, device=bboxes.device
        )

        # Concatenate image index with bbox coordinates
        rois = torch.cat([img_inds, bboxes], dim=-1)
        rois_list.append(rois)

    # Concatenate all ROIs into single tensor
    rois = torch.cat(rois_list, 0)
    return rois


def unpack_gt_instances(batch_data_samples: List) -> tuple:
    """Unpack ``gt_instances``, ``gt_instances_ignore`` and ``img_metas`` based
    on ``batch_data_samples``

    Args:
        batch_data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

    Returns:
        tuple:

            - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            - batch_gt_instances_ignore (list[:obj:`InstanceData`]):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            - batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
    """
    batch_gt_instances = []
    batch_gt_instances_ignore = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
        batch_gt_instances.append(data_sample.gt_instances)
        if "ignored_instances" in data_sample:
            batch_gt_instances_ignore.append(data_sample.ignored_instances)
        else:
            batch_gt_instances_ignore.append(None)

    return batch_gt_instances, batch_gt_instances_ignore, batch_img_metas