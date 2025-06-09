import os
from segment_anything.utils.transforms import ResizeLongestSide
import torch

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All')}")
from model.utils import load_config
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from model.mask.mask2former import add_maskformer2_config
from model.mask.predictor import Mask2FormerRunner
import numpy as np
from model.sam import GSamModel
from exps.exp_dataset import VividDataset

# from transformers.models.sam.modeling_sam import (
#     SamModel,
#     SamPreTrainedModel,
#     SamVisionEncoder,
#     SamMaskDecoder,
#     SamPositionalEmbedding,
#     SamPromptEncoder,
#     SamVisionLayer,
#     SamVisionConfig,
#     SamVisionEncoderOutput,
# )
import torchvision.transforms as transforms
from PIL import Image
from mask_pipeline_utils import _compute_points_from_mask, _compute_box_from_mask
from transformers import SamProcessor

import torch
import torchvision
import torchshow


def visualize_and_save(
    mask_instance, box, point_coords, point_labels, save_path="./show.png"
):
    """
    Visualizes bounding boxes and keypoints on a given tensor image and saves the output.

    Parameters:
    - mask_instance (torch.Tensor): The input tensor image of shape (H, W).
    - box (list or torch.Tensor): Bounding box coordinates [x_min, y_min, x_max, y_max].
    - point_coords (torch.Tensor): Coordinates of points, shape (num_points, 2).
    - point_labels (torch.Tensor): Labels for points (0 or 1).
    - save_path (str): Path to save the output image.

    Returns:
    - None (saves the image to the specified path)
    """

    # Draw bounding box
    tensor_with_bboxes = torchvision.utils.draw_bounding_boxes(
        mask_instance.unsqueeze(0),  # Convert to (1, H, W) for torchvision
        torch.tensor(box).unsqueeze(0),  # Convert to (1, 4) format
        labels=["a"],
        colors=["red"],
        width=4,
    )

    # Get points with labels 0 and 1
    point_label_0 = point_coords[point_labels == 0]
    point_label_1 = point_coords[point_labels == 1]

    # Draw keypoints for label 0 (green)
    tensor_with_keypoints = torchvision.utils.draw_keypoints(
        tensor_with_bboxes,
        torch.tensor(point_label_0).unsqueeze(0),  # Convert to (1, num_points, 2)
        colors="green",
        radius=10,
    )

    # Draw keypoints for label 1 (red)
    tensor_with_keypoints = torchvision.utils.draw_keypoints(
        tensor_with_keypoints,
        torch.tensor(point_label_1).unsqueeze(0),  # Convert to (1, num_points, 2)
        colors="red",
        radius=10,
    )

    # Save the image
    torchshow.save(tensor_with_keypoints, save_path)


class MaskPipeline:
    def __init__(
        self,
        devices="cpu",
        mask_ckpt="checkpoints/mask_model.pth",
        sam_id="/data/models/sam/huggingface/sam-vit-huge/",
    ):
        self.devices = devices
        # Load config file and build models
        self.cfg = load_config("config/prompter_aruix.yaml")
        self.sam_processor = SamProcessor.from_pretrained(sam_id)
        self.sam_model = GSamModel.from_pretrained(sam_id).to(self.devices)
        self.vision_encoder = self.sam_model.vision_encoder.to(self.devices)
        self.vision_encoder.eval()
        self.mask_decoder = self.sam_model.mask_decoder.to(self.devices)
        self.mask_decoder.eval()
        self.sam_model.eval()
        # mask2former
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(
            "config/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
        )
        cfg.merge_from_list(["MODEL.WEIGHTS", mask_ckpt])
        cfg.freeze()
        self.mask2former = Mask2FormerRunner(cfg)

        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def resize_and_pad(self, feat, expected_shape):
        """
        Resizes and pads logits to the expected shape.

        Parameters:
        logits (numpy.ndarray): Input array to be resized and padded.
        expected_shape (tuple): Expected (height, width) shape.

        Returns:
        numpy.ndarray: Resized and padded logits.
        """
        if feat.shape[0] == feat.shape[1]:  # shape is square
            trafo = ResizeLongestSide(expected_shape[0])
            feat = trafo.apply_image(feat)
        else:  # shape is not square
            trafo = ResizeLongestSide(expected_shape[0])
            feat = trafo.apply_image(feat)

            # Pad the other side
            h, w, _ = feat.shape
            padh = expected_shape[0] - h
            padw = expected_shape[1] - w

            # IMPORTANT: need to pad with zero, otherwise SAM doesn't understand the padding
            pad_width = ((0, padh), (0, padw), (0, 0))
            feat = np.pad(feat, pad_width, mode="constant", constant_values=0)
        return feat

    # brought from https://github.com/computational-cell-analytics/micro-sam/blob/83997ff4a471cd2159fda4e26d1445f3be79eb08/micro_sam/prompt_based_segmentation.py#L71

    def __call__(self, img_path):
        with open(img_path, "rb") as f:
            image = Image.open(f)
            mask2former_img = np.array(image)[:, :, ::-1]  # mask2former need BGR
            sam_img = np.array(image)

        mask_coarse, visualized_output = self.mask2former.run_on_image(mask2former_img)
        # mask_coarse_merged = mask_coarse.type(torch.bool).any(dim=0)
        final_masks = []
        for mask_instance in mask_coarse:

            ## get fine mask
            ## using hf way
            point_coords, point_labels = _compute_points_from_mask(
                mask_instance.bool().numpy(),
                original_size=None,
                box_extension=0.0,
            )
            point_coords, point_labels = (
                point_coords.astype(int).tolist(),
                point_labels.tolist(),
            )
            box = _compute_box_from_mask(
                mask_instance.bool().numpy(),
                original_size=None,
                box_extension=0.0,
            )

            # vis the prompt
            # visualize_and_save(mask_instance, box, point_coords, point_labels)
            inputs = self.sam_processor(
                images=sam_img,
                input_boxes=[[box.astype(float).tolist()]],
                input_points=[point_coords],
                input_labels=[point_labels],
                segmentation_maps=mask_instance,
                return_tensors="pt",
            )

            results = self.sam_model(**inputs, input_masks=inputs["labels"].float())
            masks = self.sam_processor.post_process_masks(
                results.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"],
            )
            # find the highest iou_scores

            # results['iou_scores']
            highest_iou_idx = torch.argmax(results["iou_scores"])

            final_mask = masks[0][0][highest_iou_idx]
            final_masks.append(final_mask)

        final_masks = torch.stack(final_masks)
        return final_masks


def evaluate_vivid():
    """
    Evaluate the Vivid dataset using the MaskPipeline and MeanAveragePrecision metric.
    This function processes images from the Vivid dataset, applies the MaskPipeline to generate predicted masks,
    and computes the mean average precision (mAP) for the segmentation task.
    It uses the `MeanAveragePrecision` metric from `torchmetrics` to evaluate the performance of the model.
    """
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from tqdm import tqdm

    metric = MeanAveragePrecision(iou_type="segm")
    data_root = "/data/datasets/grape/Vivid/"
    vivid_exp_dataset = VividDataset(
        data_root=data_root,
        txt_path=data_root + "anns/test.txt",
        json_path=data_root + "anns/instances_default_v4.json",
    )

    # img_paths = [
    #     "/data/datasets/grape/Vivid/imgs/777.png",
    #     # "/data/datasets/grape/Vivid/images/778.png",
    # ]
    mask_pipeline = MaskPipeline()
    cnt = 0
    print("dataset:", len(vivid_exp_dataset))
    for img_path, bboxes, gt_masks in tqdm(vivid_exp_dataset):
        try:

            pred_masks = mask_pipeline(img_path)
            if pred_masks == None:
                continue
            preds = [
                dict(
                    masks=pred_masks.any(dim=0).unsqueeze(0),
                    scores=torch.tensor([1.0]),
                    labels=torch.tensor([0]),
                )
            ]

            gts = [
                dict(
                    masks=torch.tensor(gt_masks)
                    .type(torch.bool)
                    .any(dim=0)
                    .unsqueeze(0),
                    labels=torch.tensor([0]),
                )
            ]
            metric.update(preds, gts)
            cnt += 1
            if cnt == 100:
                break
        except Exception as e:
            print(e)
            continue

    result = metric.compute()
    print("AP:", result)
    print("cnt:", cnt)


if __name__ == "__main__":
    # get the input folder and save the output folder from the command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Mask Pipeline for Vivid Dataset")
    parser.add_argument(
        "--img-dir",
        type=str,
        default="/data/datasets/grape/Vivid/imgs/",
        help="Path to the input folder containing images",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./output/",
        help="Path to the folder where output images will be saved",
    )
    parser.add_argument(
        "--mask-ckpt",
        type=str,
        default="checkpoints/mask_model.pth",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--sam-pth",
        type=str,
        default="facebook/sam-vit-huge",
        help="Path/name to the sam model checkpoint file",
    )
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    # Initialize the MaskPipeline
    mask_pipeline = MaskPipeline(
        devices="cuda" if torch.cuda.is_available() else "cpu",
        mask_ckpt=args.mask_ckpt,
        sam_id=args.sam_pth,
    )
    # Process images in the input directory
    for img_file in os.listdir(args.img_dir):
        if img_file.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(args.img_dir, img_file)
            print(f"Processing {img_path}...")
            pred_masks = mask_pipeline(img_path)
            if pred_masks is not None:
                # Save the predicted masks or visualize them
                save_path = os.path.join(args.save_dir, f"pred_{img_file}")
                visualize_and_save(
                    pred_masks.any(dim=0).cpu(),
                    [0, 0, pred_masks.shape[2], pred_masks.shape[1]],
                    torch.tensor([[0, 0]]),
                    torch.tensor([0]),
                    save_path=save_path,
                )
            else:
                print(f"No masks predicted for {img_path}.")

