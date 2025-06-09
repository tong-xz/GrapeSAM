from copy import copy
import cv2
from matplotlib import cm
import torch
import os
import numpy as np
from .models.vgg import vgg19
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
from torchvision import transforms
from PIL import Image

args = None


def show_heatmap(heatmap):
    heatmap = heatmap.squeeze(0).squeeze(0).cpu().numpy()
    # heatmap shaped as (H, W)
    # erosion by using cv2
    kernel = np.ones((3, 3), np.uint8)
    heatmap = cv2.dilate(heatmap, kernel, iterations=2)

    heatmap[heatmap < 0.1] = np.nan
    plt.imshow(heatmap, cmap="Reds", interpolation="nearest")
    # plt.colorbar()
    plt.axis("off")
    plt.savefig("heatmap_wo_bkgd.png", bbox_inches="tight", pad_inches=0.1, dpi=400)
    plt.close()


def show_heatmap_on_raw_image(heatmap, img):
    import cv2
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 10))
    # make raw image from tensor to numpy array
    if torch.is_tensor(img):
        img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    # resize raw image to match heatmap size
    img = cv2.resize(
        img, (heatmap.shape[3], heatmap.shape[2]), interpolation=cv2.INTER_LINEAR
    )
    # normalize heatmap from 0 to 1
    heatmap = heatmap.squeeze(0).squeeze(0).cpu().numpy()
    # erosion by using cv2
    kernel = np.ones((3, 3), np.uint8)
    heatmap = cv2.dilate(heatmap, kernel, iterations=2)
    heatmap[heatmap < 0.1] = np.nan
    plt.imshow(img, interpolation="nearest", alpha=0.6)
    plt.imshow(heatmap, cmap="Reds", interpolation="nearest", alpha=0.6)
    # plt.colorbar()
    plt.axis("off")
    plt.savefig("heatmap_w_bkgd.png", bbox_inches="tight", pad_inches=0.1, dpi=400)
    plt.close()


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

    pred_points_sam = [[[point] for point in pred_points[0]]]

    return pred_points_sam, pred_points_score


def sam_points_inference(
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


class PointModel:
    def __init__(self, model_path, device="cuda"):
        self.model = vgg19().eval().to(device)
        # self.model.load_state_dict(torch.load(model_path, device)["model_state_dict"])
        self.model.load_state_dict(torch.load(model_path, device))
        self.device = device

    def __call__(self, img):
        self.model.eval()  # Set model to evaluation mode
        try:
            if not isinstance(img, torch.Tensor):
                raise TypeError("Input must be a torch.Tensor")
            if img.device != self.device:
                img = img.to(self.device)
            with torch.no_grad():
                heatmap = self.model(img)
                # breakpoint()
                pred_points, pred_points_score = convert_heatmap_to_points(
                    heatmap, point_threshold=0.05
                )
                return pred_points, pred_points_score

        except Exception as e:
            print(f"Error during model inference: {str(e)}")
            raise


def parse_args():
    parser = argparse.ArgumentParser(description="Test ")
    parser.add_argument(
        "--data-dir", default="../../data/UCF_Bayes", help="training data directory"
    )
    parser.add_argument("--save-dir", default="./model.pth", help="model path")
    parser.add_argument("--device", default="0", help="assign device")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.strip()  # set vis gpu
    # image preparation
    # img_path = "/home/xz/Dev/baseline-exp-playground/DATASET/vivid-close/test/1084.png"
    img_path = "/home/xz/Pictures/9.jpg"
    img = Image.open(img_path).convert("RGB")

    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = trans(img).unsqueeze(0).to("cuda")

    # model preparation

    point_model = PointModel(args.save_dir)

    # segment anything
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    pred_points, pred_points_score = point_model(img_tensor)

    scaled_img, scaled_keypoints = scale_image_and_keypoints(img, pred_points)
    # visualize_image_and_keypoints(scaled_img, scaled_keypoints)

    best_masks, best_scores = sam_points_inference(
        sam_model,
        sam_processor,
        scaled_img,
        scaled_keypoints,
        optimal=True,
        multimask_output=True,
    )

    show_masks_on_image(
        scaled_img, best_masks, title="6-better.png", alpha=0.7, show_background=False
    )

# python3 build.py --data-dir /home/xz/Dev/baseline-exp-playground/DATASET/vivid-close --save-dir /home/xz/Dev/GrapeSAM/point/output/0125-224233/best_val.pth
