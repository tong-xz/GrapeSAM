import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
from torchvision import transforms
from PIL import Image
from utils.show import show_masks_on_image

args = None


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[
        1
    ]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes


def _nms(heat, kernel):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def convert_heatmap_to_points(
    outputs, nms_kernel_size=3, point_threshold=0.05, max_points=1024
):
    outputs = F.interpolate(outputs, scale_factor=8, mode="bilinear")
    pred_heatmaps_nms = _nms(outputs.detach().clone(), nms_kernel_size)
    pred_points, pred_points_score = (
        torch.zeros(1, max_points, 2).cuda(),
        torch.zeros(1, max_points).cuda(),
    )
    m = 0
    for i in range(1):  # since batch size is 1
        points = torch.nonzero((pred_heatmaps_nms[i] > point_threshold).squeeze())
        points = torch.flip(points, dims=(-1,))
        pred_points_score_ = pred_heatmaps_nms[
            i, 0, points[:, 1], points[:, 0]
        ].flatten(0)

        idx = torch.argsort(pred_points_score_, dim=0, descending=True)[
            : min(max_points, pred_points_score_.size(0))
        ]
        points = points[idx]
        pred_points_score_ = pred_points_score_[idx]

        pred_points[i, : points.size(0)] = points
        pred_points_score[i, : points.size(0)] = pred_points_score_
        m = max(m, points.size(0))

    pred_points = pred_points[:, :m]
    pred_points_score = pred_points_score[:, :m]

    # prepare for sam inference format
    pred_points = pred_points.cpu().tolist()

    # pred_points_sam = [[[point] for point in pred_points[0]]]

    return pred_points, pred_points_score


def visualize_image(tensor_image, heatmap=None, points=None):
    """
    Visualize a normal map tensor, optionally a heatmap, and scatter points.

    :param tensor_image: Tensor of shape (1, 3, H, W)
    :param heatmap: Optional heatmap tensor of shape (1, H, W)
    :param points: Optional points tensor of shape (1, n, 2) for scatter plot
    """
    # Ensure the tensor is on the CPU and convert it to a numpy array
    img_np = tensor_image.squeeze().permute(1, 2, 0).cpu().numpy()

    # De-normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)  # Clip values to [0, 1]

    # Determine the number of plots
    num_plots = 1 + (1 if heatmap is not None else 0)
    fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axs = [axs]

    # Plot the image
    axs[0].imshow(img_np)
    axs[0].set_title("Normal Map")
    axs[0].axis("off")  # Hide the axis

    # Plot points if provided
    if points is not None:
        points_np = points.squeeze().cpu().numpy()
        axs[0].scatter(points_np[:, 0], points_np[:, 1], c="red", s=1)

    # Plot the heatmap if provided
    if heatmap is not None:
        heatmap_np = heatmap.squeeze().cpu().numpy()
        axs[1].imshow(heatmap_np, cmap="hot")
        axs[1].set_title("Heatmap")
        axs[1].axis("off")  # Hide the axis
        if points is not None:
            axs[1].scatter(points_np[:, 0], points_np[:, 1], c="red", s=1)

    plt.show()


def predict_points(model, img): ...


def sam_points_inference(model, processor, raw_image, points, multimask_output):
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
    return masks, scores


def parse_args():
    parser = argparse.ArgumentParser(description="Test ")
    parser.add_argument(
        "--data-dir", default="../../data/UCF_Bayes", help="training data directory"
    )
    parser.add_argument("--save-dir", default="./model.pth", help="model path")
    parser.add_argument("--device", default="0", help="assign device")
    args = parser.parse_args()
    return args


def resize_and_pad(img, target_size=1024):
    """
    Resize image to fit within target_size x target_size while maintaining aspect ratio,
    then pad with black to make it exactly target_size x target_size.

    Args:
        img: PIL Image or tensor of shape (C, H, W)
        target_size: desired size (both height and width)
    Returns:
        Padded tensor of shape (C, target_size, target_size)
    """
    # Calculate scaling factor to fit within target size
    width, height = img.size
    scale = min(target_size / width, target_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create black background
    background = Image.new("RGB", (target_size, target_size), (0, 0, 0))

    # Paste resized image in center
    offset_x = (target_size - new_width) // 2
    offset_y = (target_size - new_height) // 2
    background.paste(img_resized, (offset_x, offset_y))

    # Convert back to tensor and normalize
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return trans(background).unsqueeze(0)


def scale_image_and_keypoints(img, keypoints, target_size=(1024, 1024)):
    """
    将图片缩放到指定大小并保持比例，同时相应地缩放关键点坐标

    Args:
        img: PIL Image对象 (RGB)
        keypoints: numpy array，形状为(N, 3)，表示点的坐标和尺度 [[x1,y1,s1], [x2,y2,s2], ...]
        target_size: tuple，目标尺寸 (width, height)

    Returns:
        scaled_img: 缩放后的RGB图片
        scaled_keypoints: 缩放后的关键点坐标
    """
    # 获取原始尺寸
    orig_w, orig_h = img.size
    target_w, target_h = target_size

    # 计算缩放比例新的尺寸
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # 缩放图片
    scaled_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 创建新的RGB背景
    final_img = Image.new("RGB", target_size, (0, 0, 0))

    # 计算粘贴位置（居中）
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    final_img.paste(scaled_img, (paste_x, paste_y))

    # Convert keypoints list to numpy array
    keypoints = np.array(keypoints[0])  # keypoints[0] since it's a nested list
    scaled_keypoints = keypoints.copy()

    scaled_keypoints[:, 0] = keypoints[:, 0] * scale + paste_x
    scaled_keypoints[:, 1] = keypoints[:, 1] * scale + paste_y

    scaled_keypoints = scaled_keypoints.tolist()
    scaled_keypoints = [[[point] for point in scaled_keypoints]]

    # [[[point] for point in pred_points[0]]]

    return final_img, scaled_keypoints
    return final_img, scaled_keypoints


def visualize_image_and_keypoints(img, keypoints, title="Image with Keypoints"):
    """
    可视化图片和关键点

    Args:
        img: PIL Image对象
        keypoints: numpy array，关键点坐标
        title: 显示的标题
    """
    import matplotlib.pyplot as plt

    # 转换PIL图像为numpy数组用于matplotlib显示
    img_array = np.array(img)

    plt.figure(figsize=(12, 12))
    plt.imshow(img_array)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c="red", s=50)
    plt.title(title)
    plt.axis("on")
    plt.show()


def build_point_model(path, device="cuda"):
    model = vgg19().to(device)
    model.load_state_dict(torch.load(path, device))
    return model


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.strip()  # set vis gpu
    # image preparation
    # img_path = "/home/xz/Dev/baseline-exp-playground/DATASET/vivid-close/test/1084.png"
    img_path = "/home/xz/Pictures/5.jpg"
    img = Image.open(img_path).convert("RGB")

    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = trans(img).unsqueeze(0).to("cuda")

    # model preparation
    point_model = build_point_model(args.save_dir)

    # segment anything
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    with torch.no_grad():
        heatmap = point_model(img_tensor)

        # Convert outputs to points
        pred_points, pred_points_score = convert_heatmap_to_points(heatmap)

        scaled_img, scaled_keypoints = scale_image_and_keypoints(img, pred_points)
        # visualize_image_and_keypoints(scaled_img, scaled_keypoints)

        # convert to 1024.

        masks, scores = sam_points_inference(
            sam_model,
            sam_processor,
            scaled_img,
            scaled_keypoints,
            multimask_output=True,
        )

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

        show_masks_on_image(
            scaled_img, best_masks, best_scores, title="5-better.png", alpha=0.8
        )

# python3 build.py --data-dir /home/xz/Dev/baseline-exp-playground/DATASET/vivid-close --save-dir /home/xz/Dev/GrapeSAM/point/output/0125-224233/best_val.pth
