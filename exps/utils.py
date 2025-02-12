from matplotlib import pyplot as plt
import numpy as np
import torch


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color is not None:
        color = np.array(color)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.show()


def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis("on")
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_masks_on_image(raw_image, masks, scores=None, title=None, transparency=0.8):
    # Handle single mask case
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if len(masks.shape) == 2:  # Single mask
        masks = masks[None, ...]  # Add batch dimension

    # Handle single score case
    if isinstance(scores, torch.Tensor):
        if scores.shape[0] == 1:
            scores = scores.squeeze()
        if scores.ndim == 0:  # Single score
            scores = scores[None]  # Add batch dimension

    # Create a single subplot
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(raw_image))

    # 预定义一组明亮的颜色
    BRIGHT_COLORS = [
        [1.0, 0.0, 0.0],  # 红
        [0.0, 1.0, 0.0],  # 绿
        [0.0, 0.0, 1.0],  # 蓝
        [1.0, 1.0, 0.0],  # 黄
        [1.0, 0.0, 1.0],  # 品红
        [0.0, 1.0, 1.0],  # 青
    ]

    # Show all masks on the same image with bright colors
    for i, mask in enumerate(masks):
        mask = mask.cpu().detach()
        color = BRIGHT_COLORS[i % len(BRIGHT_COLORS)] + [transparency]
        show_mask(mask, plt.gca(), random_color=False, color=color)
    # Add a title showing all scores
    # score_text = "\n".join(
    #     [f"Mask {i+1} Score: {score.item():.3f}" for i, score in enumerate(scores)]
    # )
    plt.title(title)
    plt.axis("off")
    plt.savefig(f"{title}.png")
    plt.show()
