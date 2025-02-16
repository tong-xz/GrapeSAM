from model import PointModel, utils, sam
from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import SamModel, SamProcessor, pipeline
import os
from tqdm import tqdm
import time
import gc
import psutil  # Add this import at the top
import argparse
import numpy as np
from model.utils import load_config, show_masks_on_image
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from model.mask.mask2former import add_maskformer2_config
from model.mask.predictor import Mask2FormerRunner
import torchshow


class GrapePipeline:
    def __init__(
        self,
        point_model_path,
        mask_ckpt,
        img_save_path,
        mask_cfg="config/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        sam_from_pretrained="facebook/sam-vit-huge",
    ) -> None:
        """Initialize the grape detection pipeline with required models and configurations.

        Args:
            point_model_path (str): Path to the pretrained point detection model checkpoint
            img_save_path (str): Directory path where output images will be saved
        """
        # Print device information at initialization
        self._print_device_info()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nInitializing models on: {self.device}")

        self.sam_model = SamModel.from_pretrained(sam_from_pretrained).to(self.device)
        self.sam_processor = SamProcessor.from_pretrained(sam_from_pretrained)
        self.generator = pipeline(
            "mask-generation",
            model="facebook/sam-vit-huge",
            image_processor=self.sam_processor.image_processor,
            device=self.device,
        )

        # berry part
        self.point_model = PointModel(point_model_path, self.device)
        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.img_save_path = img_save_path
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)

        # grape cluster part
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(mask_cfg)
        cfg.merge_from_list(["MODEL.WEIGHTS", mask_ckpt])
        cfg.freeze()
        self.mask2former = Mask2FormerRunner(cfg)

    @staticmethod
    def _print_device_info():
        """Print detailed information about the computing device (GPU/CPU) being used."""
        print("\n=== Device Information ===")

        # Check CUDA availability and visible devices
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All')}")

        # Get the current device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Current device: {device}")

        if device.type == "cuda":
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(
                    f"  Memory: {props.total_memory/1024**3:.1f}GB total, "
                    f"{torch.cuda.memory_allocated(i)/1024**3:.1f}GB used, "
                    f"{torch.cuda.memory_reserved(i)/1024**3:.1f}GB cached"
                )
                print(f"  CUDA: {props.major}.{props.minor}")
        else:
            import platform, psutil

            print("\nCPU Info:")
            print(f"  {platform.processor()}")
            print(
                f"  Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} total"
            )
            print(
                f"  Memory: {psutil.virtual_memory().total/1024**3:.1f}GB total, "
                f"{psutil.virtual_memory().available/1024**3:.1f}GB free"
            )

    def _resize_img(self, img_path):
        """Resize input image to match SAM processor's target dimensions.

        Args:
            img_path (str): Path to the input image file

        Returns:
            PIL.Image: Resized image in PIL format matching SAM's required dimensions
        """
        raw_img = Image.open(img_path).convert("RGB")
        inputs = self.sam_processor(images=raw_img, return_tensors="pt").to("cuda")
        target_shape = inputs["reshaped_input_sizes"][0]
        target_height, target_width = target_shape[0], target_shape[1]

        # Resize image to match target dimensions
        if raw_img.size != (
            target_width,
            target_height,
        ):  # PIL uses (width, height) order
            raw_img = raw_img.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )
        return raw_img

    def segment_grape_cluster(self, img_path):
        """Segment grape clusters in the input image.

        Args:
            img_path (str): Path to the input image file

        Returns:
            tuple: Segmentation results for grape clusters (implementation pending)
        """
        with open(img_path, "rb") as f:
            image = Image.open(f)
            mask2former_img = np.array(image)[:, :, ::-1]  # mask2former need BGR

        mask_instance, visualized_output = self.mask2former.run_on_image(
            mask2former_img
        )

        return mask_instance

    def segment_berry(self, img_path):
        """Segment individual berries in a grape image using point detection model and SAM.

        Args:
            img_path (str): Path to the input image file

        Returns:
            tuple: A tuple containing:
                - PIL.Image: The original input image
                - torch.Tensor: Binary masks for detected berries (N x 1 x H x W)
                - torch.Tensor: Confidence scores for each detected berry mask
        """
        best_masks_cpu = None
        best_scores_cpu = None

        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.trans(img).unsqueeze(0).to(self.device)
            # scatter point prediction
            pred_points, pred_points_score = self.point_model(img_tensor)
            # sam segment by points
            best_masks, best_scores = sam.predict_by_points(
                self.sam_model,
                self.sam_processor,
                img,
                pred_points,
                optimal=True,
                multimask_output=True,
            )
            # Create copies on CPU before cleanup
            best_masks_cpu = best_masks.cpu().detach()
            best_scores_cpu = best_scores.cpu().detach()

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
        finally:
            # Release GPU memory & CPU memory
            torch.cuda.empty_cache()
            gc.collect()

        return img, best_masks_cpu, best_scores_cpu

    def segment_everything(self, img_path, points_per_batch=256):
        """Generate segmentation masks for all objects in the image using SAM.

        Args:
            img_path (str): Path to the input image file
            points_per_batch (int, optional): Number of points to process per batch. Defaults to 256.

        Returns:
            tuple: A tuple containing:
                - PIL.Image: The resized input image
                - torch.Tensor: Binary masks for all detected objects (N x 1 x H x W)
                - torch.Tensor: Confidence scores for each mask
        """
        resized_img = self._resize_img(img_path)
        everything_masks_cpu = None
        everything_scores_cpu = None
        try:
            outputs = self.generator(resized_img, points_per_batch=points_per_batch)
            # Move masks to CPU and convert to float32
            everything_masks = torch.as_tensor(
                np.array(outputs["masks"], dtype=np.float32), device="cpu"
            )
            everything_scores = torch.as_tensor(
                np.array(outputs["scores"], dtype=np.float32), device="cpu"
            )

            everything_masks = everything_masks.unsqueeze(1)

            # Create copy on CPU before cleanup
            everything_masks_cpu = everything_masks.cpu().detach()
            everything_scores_cpu = everything_scores.cpu().detach()

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
        finally:
            # Release GPU memory & CPU memory
            torch.cuda.empty_cache()
            gc.collect()
        return resized_img, everything_masks_cpu, everything_scores_cpu

    def _group_small_masks_by_instance(self, large_masks, small_masks, threshold=0.5):
        """
        Groups small masks based on their overlap with large masks.

        Args:
            large_masks (torch.Tensor): A tensor of shape (N, H, W), where N is the number of large masks.
            small_masks (torch.Tensor): A tensor of shape (M, H, W), where M is the number of small masks.
            threshold (float): A float between 0 and 1, representing the minimum overlap percentage required for a small mask to be considered valid.

        Returns:
            list of lists: A list of N lists, where each sublist contains small masks that overlap with the corresponding large mask instance.
        """

        # Initialize a list of N empty lists (one for each large mask instance)
        grouped_masks = [[] for _ in range(large_masks.shape[0])]

        # Iterate over each small mask
        for j in range(small_masks.shape[0]):  # M small masks
            small_mask = small_masks[j]

            # Compute the overlap with all large masks (N masks)
            for i in range(large_masks.shape[0]):  # N large masks
                large_mask = large_masks[i]

                # Calculate overlap by element-wise multiplication
                overlap = large_mask * small_mask  # (H, W)

                # Calculate the percentage of overlap
                overlap_area = (
                    overlap.sum().item()
                )  # Sum of non-zero elements in the overlap
                total_area = small_mask.sum()  # Total number of elements (H * W)
                overlap_percentage = overlap_area / total_area

                # If the overlap percentage is above the threshold, group this small mask with the large mask instance
                if overlap_percentage >= threshold:
                    grouped_masks[i].append(small_mask)
                    break  # We found the large mask instance for this small mask, no need to check other large masks

        return grouped_masks

    def union_berries(m_cluster, m_berry, m_everything):
        """Post-process berry segmentation masks using cluster and everything masks.

        Args:
            m_cluster (torch.Tensor): Grape cluster segmentation masks
            m_berry (torch.Tensor): Berry segmentation masks
            m_everything (torch.Tensor): Everything segmentation masks from SAM' everything mode

        Returns:
            torch.Tensor: Filtered and refined berry segmentation masks
        """
        ...

    def cal_closure(berry_masks, cluster_masks):
        for cluster_mask in cluster_masks:
            # sum corresponding berry masks
            pass
        ...

    #  a csv file that list image name, cluster closure, berry number.
    def process_folder(self, input_folder, format):
        """Process all images in the input folder through the grape detection pipeline.

        Args:
            input_folder (str): Path to the folder containing input images
            format (str or list): File extension(s) of images to process (e.g., 'png', ['jpg', 'jpeg'])
        """
        # Get list of files with matching format first
        image_files = [f for f in os.listdir(input_folder) if f.endswith(tuple(format))]
        pbar = tqdm(total=len(image_files), desc="Processing images", unit="image")
        process = psutil.Process()

        for i, filename in enumerate(image_files):
            img_path = os.path.join(input_folder, filename)

            # Step 1: get mask set M_p, M_e
            berry_img_title = os.path.splitext(filename)[0] + "_berry"
            img, berry_masks_cpu, berry_scores_cpu = self.segment_berry(img_path)

            everything_img_title = os.path.splitext(filename)[0] + "_everything"
            img, everything_masks_cpu, everything_scores_cpu = self.segment_everything(
                img_path
            )
            # print(berry_masks_cpu.shape, everything_masks_cpu.shape)

            show_masks_on_image(
                img,
                everything_masks_cpu,
                title=berry_img_title,
                alpha=0.6,
                show_background=True,
                save_path=self.img_save_path,
            )

            # Step 2: get grape cluster mask set M3
            mask_instance = self.segment_grape_cluster(img_path)
            if mask_instance is not None:
                grouped_masks = self.group_small_masks_by_instance(
                    mask_instance, berry_masks_cpu.squeeze(1), 0.5
                )

                torchshow.save(
                    torch.stack([i * 2 for i in grouped_masks])
                    .type(torch.bool)
                    .any(dim=0)
                    .unsqueeze(0),
                    "filter.png",
                )
            # STep 3: Use M3 as filter to remove outlier masks in M1, M2

            # save predicted berry masks

            # Update progress bar with memory info
            if self.device.type == "cuda":
                used_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
                total_memory = (
                    torch.cuda.get_device_properties(0).total_memory
                    / 1024
                    / 1024
                    / 1024
                )
            else:
                used_memory = process.memory_info().rss / 1024 / 1024 / 1024
                total_memory = psutil.virtual_memory().total / 1024 / 1024 / 1024

            pbar.set_postfix({"Mem": f"{used_memory:.1f}/{total_memory:.1f}GB)"})
            pbar.update(1)

        pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Grape berry detection pipeline")
    parser.add_argument(
        "--point-ckpt",
        type=str,
        required=True,
        help="Path to the point model checkpoint file",
    )
    parser.add_argument(
        "--mask-ckpt",
        type=str,
        required=True,
        help="Path to the mask model checkpoint file",
    )
    parser.add_argument(
        "--sam-pth",
        type=str,
        default="facebook/sam-vit-huge",
        help="Path/name to the sam model checkpoint file",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input image folder"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output folder"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        help="Image format to process (default: png)",
    )

    args = parser.parse_args()

    grape_pipeline = GrapePipeline(
        args.point_ckpt, args.mask_ckpt, args.output, sam_from_pretrained=args.sam_pth
    )
    grape_pipeline.process_folder(args.input, args.format)


"""
python3 pipeline.py --point-ckpt /home/xz/Dev/baseline-exp-playground/GeneralizedLoss-Counting-Pytorch/output/0203-202057/best_val.pth \
--input /home/xz/Downloads/feb-test/converted/ \
--output /home/xz/Downloads/feb-test/converted/berry/
"""

if __name__ == "__main__":
    main()
