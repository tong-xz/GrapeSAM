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


class GrapePipeline:
    def __init__(self, point_model_path, img_save_path) -> None:
        """Initialize the grape detection pipeline with required models and configurations.

        Args:
            point_model_path (str): Path to the pretrained point detection model checkpoint
            img_save_path (str): Directory path where output images will be saved
        """
        # Print device information at initialization
        self._print_device_info()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nInitializing models on: {self.device}")

        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(
            self.device
        )
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.generator = pipeline(
            "mask-generation", model="facebook/sam-vit-huge", device=self.device
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
        # self.cfg = load_config("config/prompter_aruix.yaml")
        # cfg = get_cfg()
        # add_deeplab_config(cfg)
        # add_maskformer2_config(cfg)
        # cfg.merge_from_file(
        #     "config/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
        # )
        # cfg.merge_from_list(
        #     [
        #         "MODEL.WEIGHTS",
        #         "/data/Hypothesis/proposition/Mask2Former/output/model_final.pth",
        #     ]
        # )
        # cfg.freeze()
        # self.mask2former = Mask2FormerRunner(cfg)

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
        ...

    # add everything mode to post processing mask
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

    def post_process_berries(m_cluster, m_berry, m_everything):
        """Post-process berry segmentation masks using cluster and everything masks.

        Args:
            m_cluster (torch.Tensor): Grape cluster segmentation masks
            m_berry (torch.Tensor): Berry segmentation masks
            m_everything (torch.Tensor): Everything segmentation masks

        Returns:
            torch.Tensor: Filtered and refined berry segmentation masks
        """
        # s1 filter out

        # s2 superset
        ...

    def process_folder(self, input_folder, format):
        """Process all images in the input folder through the grape detection pipeline.

        Args:
            input_folder (str): Path to the folder containing input images
            format (str or list): File extension(s) of images to process (e.g., 'png', ['jpg', 'jpeg'])
        """
        # Get list of files with matching format first
        image_files = [f for f in os.listdir(input_folder) if f.endswith(tuple(format))]

        # Initialize progress bar with memory usage
        pbar = tqdm(total=len(image_files), desc="Processing images", unit="image")
        process = psutil.Process()

        for i, filename in enumerate(image_files):
            # Get memory usage before processing
            if self.device.type == "cuda":
                mem_before = (
                    torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                )  # Convert to GB
            else:
                mem_before = (
                    process.memory_info().rss / 1024 / 1024 / 1024
                )  # Convert to GB

            img_path = os.path.join(input_folder, filename)

            # Step 1: get mask set M_p, M_e
            berry_img_title = os.path.splitext(filename)[0] + "_berry"
            img, berry_masks_cpu, _ = self.segment_berry(img_path)

            everything_img_title = os.path.splitext(filename)[0] + "_everything"
            img, everything_masks_cpu, everything_scores_cpu = self.segment_everything(
                img_path
            )
            # print(berry_masks_cpu.shape, everything_masks_cpu.shape)

            show_masks_on_image(
                img,
                berry_masks_cpu,
                title=berry_img_title,
                alpha=0.6,
                show_background=True,
                save_path=self.img_save_path,
            )

            # Step 2: get grape cluster mask set M3

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

            pbar.set_postfix(
                {
                    "Mem": f"{used_memory:.1f}/{total_memory:.1f}GB ({used_memory/total_memory:.1%})"
                }
            )
            pbar.update(1)

        pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Grape berry detection pipeline")
    parser.add_argument(
        "--point-ckpt",
        type=str,
        required=True,
        help="Path to the point modelcheckpoint file",
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
        args.point_ckpt,
        args.output,
    )
    grape_pipeline.process_folder(args.input, args.format)


"""
python3 pipeline.py --point-ckpt /home/xz/Dev/baseline-exp-playground/GeneralizedLoss-Counting-Pytorch/output/0203-202057/best_val.pth \
--input /home/xz/Downloads/feb-test/converted/ \
--output /home/xz/Downloads/feb-test/converted/berry/
"""
if __name__ == "__main__":
    main()
