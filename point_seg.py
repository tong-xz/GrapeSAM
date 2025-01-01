# build instance segmentation pure by points

from model import build_loader
from model.utils import load_config
from model import sam_hf
import torch
from transformers import SamModel
from model.sam_hf import GSamModel
import numpy as np
import matplotlib.pyplot as plt

CONFIG_PATH = "/home/xz/Dev/GrapeSAM/config/prompter_huge.yaml"
ROOT_DIR = "/home/xz/Dev/GrapeSAM/data/vivid"


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def visualize_attention(image, attention_maps, save_path=None):
    """
    Visualize attention maps from SAM model

    Args:
        image: Input image tensor of shape (C, H, W)
        attention_maps: Attention weights from vision encoder
        save_path: Optional path to save visualization
    """
    # Convert image to numpy and normalize to [0,1]
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # Get attention maps from the last layer (usually most informative)
    # Shape: (num_heads, h*w, h*w)
    last_layer_attention = attention_maps[-1][0]  # Take first batch
    num_heads = last_layer_attention.shape[0]

    # Create subplot grid
    fig, axes = plt.subplots(2, num_heads // 2, figsize=(15, 8))
    axes = axes.ravel()

    for head_idx in range(num_heads):
        ax = axes[head_idx]

        # Reshape attention map to spatial dimensions
        attn_map = last_layer_attention[head_idx].mean(
            dim=0
        )  # Average over source tokens
        size = int(np.sqrt(attn_map.shape[0]))
        attn_map = attn_map.reshape(size, size).cpu().numpy()

        # Overlay attention map on image
        ax.imshow(img_np)
        ax.imshow(attn_map, alpha=0.5, cmap="hot")
        ax.axis("off")
        ax.set_title(f"Head {head_idx+1}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


cfg = load_config(CONFIG_PATH)
model = GSamModel.from_pretrained("facebook/sam-vit-base").cuda()

test_loader = build_loader(root_dir=ROOT_DIR, batch_size=1)["test"]

for imgs, points in test_loader:
    imgs = imgs.to("cuda")
    points = points.unsqueeze(0).to("cuda")
    labels = torch.ones(points.shape[:-1]).to("cuda")

    with torch.no_grad():
        # Get vision encoder outputs with attention weights
        image_embeddings = model.vision_encoder(
            imgs, output_attentions=True, return_dict=True
        )
        breakpoint()
        # Visualize attention maps
        visualize_attention(
            imgs[0], image_embeddings.attentions
        )  # Take first image in batch

    #     sparse_embeddings, dense_embeddings = model.prompt_encoder(
    #         input_points=points, input_labels=labels, input_boxes=None, input_masks=None
    #     )
    #     image_positional_embeddings = model.get_image_wide_positional_embeddings()
    #     low_res_masks, iou_predictions, mask_decoder_attentions = model.mask_decoder(
    #         image_embeddings=image_embeddings,
    #         image_positional_embeddings=image_positional_embeddings,
    #         sparse_prompt_embeddings=sparse_embeddings,
    #         dense_prompt_embeddings=dense_embeddings,
    #         multimask_output=True,
    #         attention_similarity=None,
    #         target_embedding=None,
    #         output_attentions=None,
    #     )

    # # Convert masks to numpy and get IoU scores
    # masks = low_res_masks.squeeze(0).squeeze(0).cpu().numpy()  # Shape: (3, 256, 256)
    # iou_scores = iou_predictions.squeeze(0).squeeze(0).cpu().numpy()  # Shape: (3,)

    # # Create subplot for each mask
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # # Display original image
    # img_np = imgs.squeeze(0).cpu().permute(1, 2, 0).numpy()
    # for ax in axes:
    #     ax.imshow(img_np)

    # # Show masks with IoU scores as titles
    # for idx, (mask, iou, ax) in enumerate(zip(masks, iou_scores, axes)):
    #     show_mask(mask, ax, random_color=True)
    #     ax.set_title(f"Mask {idx+1}, IoU: {iou:.3f}")
    #     ax.axis("off")

    # plt.tight_layout()
    # plt.show()

    # breakpoint()
