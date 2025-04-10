from matplotlib import pyplot as plt
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
from model.utils import load_config, show_masks_on_image, show_grape_and_berry
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from model.mask.mask2former import add_maskformer2_config
from model.mask.predictor import Mask2FormerRunner
import pandas as pd
import random
import warnings


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

    def _resize_img(self, img_path, return_shapes=False):
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

        if return_shapes:
            return raw_img, inputs["original_sizes"], inputs["reshaped_input_sizes"]

        return raw_img

    def segment_grape_cluster(self, img):
        """Segment grape clusters in the input image.

        Args:
            img (PIL.Image): Input image in PIL format

        Returns:
            torch.Tensor: Binary masks for detected grape clusters (N x H x W)
                where N is the number of detected clusters
        """
        mask2former_img = np.array(img)[:, :, ::-1]  # mask2former need BGR
        mask_instance, _ = self.mask2former.run_on_image(mask2former_img)

        return mask_instance

    def segment_berry(self, img):
        """Segment individual berries in a grape image using point detection model and SAM.

        Args:
            img (PIL.Image): Input image in PIL format

        Returns:
            tuple: A tuple containing:
                - PIL.Image: The input image
                - torch.Tensor: Binary masks for detected berries (N x 1 x H x W)
                - torch.Tensor: Confidence scores for each detected berry mask (N,)
                  where N is the number of detected berries
        """

        try:
            img_tensor = self.trans(img).unsqueeze(0).to(self.device)
            pred_points, pred_points_score = self.point_model(img_tensor)
            best_masks, best_scores = sam.predict_by_points(
                self.sam_model,
                self.sam_processor,
                img,
                pred_points,
                optimal=True,
                multimask_output=True,
            )
            return img, best_masks.cpu(), best_scores.cpu()
        except Exception as e:
            print(f"Error processing: {str(e)}")
            return img, None, None

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
        resized_img, ori_shape, resized_shape = self._resize_img(
            img_path, return_shapes=True
        )
        everything_masks_cpu = None
        everything_scores_cpu = None
        try:
            outputs = self.generator(resized_img, points_per_batch=points_per_batch)
            everything_masks = torch.as_tensor(
                np.array(outputs["masks"], dtype=np.float32), device="cpu"
            )
            everything_masks = self.sam_processor.image_processor.post_process_masks(
                everything_masks.unsqueeze(0).unsqueeze(2), ori_shape, resized_shape
            )[0]
            everything_scores = torch.as_tensor(
                np.array(outputs["scores"], dtype=np.float32), device="cpu"
            )
            return resized_img, everything_masks, everything_scores
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return resized_img, None, None

    def filter_masks_by_iqr(self, masks: torch.Tensor, log=False) -> torch.Tensor:
        # Step 1: Calculate the area of each mask (sum of True values)
        areas = masks.sum(dim=(1, 2))  # Sum along H and W for each mask
        if log:
            # Avoid log(0) by adding a small constant (e.g., 1e-6) to proportions_ones
            areas = np.log(areas + 1e-6)
            # Normalize log values to [0, 1] (optional step depending on the desired scale)
            areas = (areas - areas.min()) / (areas.max() - areas.min())

        # Step 2: Calculate the 25th (Q1) and 75th (Q3) percentiles
        Q1 = areas.quantile(0.25)
        Q3 = areas.quantile(0.75)

        # Step 3: Calculate the IQR
        IQR = Q3 - Q1

        # Step 4: Calculate the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Step 5: Filter the masks based on the calculated bounds
        valid_masks = (areas >= lower_bound) & (areas <= upper_bound)

        # Step 6: Return the filtered masks (using the valid mask indices)
        return masks[valid_masks]

    def filter_masks_by_gaussian(
        self, masks: torch.Tensor, num_std_dev: float = 2.0
    ) -> torch.Tensor:
        """
        prompt: I have a set of bool masks, which is a Pytorch tensor shaped as (N, H, W),
        N is the number of masks. I want to filter out some very large or small masks.
        I cannot determine the max or min area. Please use the suitable method.
        """
        # Step 1: Calculate the area of each mask (sum of True values)
        areas = masks.sum(dim=(1, 2))  # Sum along H and W for each mask

        # Step 2: Calculate the mean and standard deviation of the areas
        mean_area = areas.mean()
        std_area = areas.std()

        # Step 3: Calculate the range based on mean ± num_std_dev * std
        lower_bound = mean_area - num_std_dev * std_area
        upper_bound = mean_area + num_std_dev * std_area

        # Step 4: Filter the masks based on the calculated area bounds
        valid_masks = (areas >= lower_bound) & (areas <= upper_bound)

        # Step 5: Return the filtered masks (using the valid mask indices)
        return masks[valid_masks]

    def check_memory_limit(self, limit_gb=60):  # Set threshold below 64GB
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info().rss  # Resident memory usage (bytes)
        mem_gb = mem_info / (1024**3)  # Convert to GB

        if mem_gb > limit_gb:
            raise MemoryError(
                f"Memory usage exceeded {limit_gb}GB! Current: {mem_gb:.2f}GB"
            )

    def _filter_abnormal_masks(self, masks, k=1):
        """
        Filters masks based on their area relative to the mean and standard deviation.

        Args:
            masks (torch.Tensor): A tensor of shape [N, H, W] where each element is 0 or 1.
            k (int, optional): The number of standard deviations from the mean to consider. Defaults to 2.

        Returns:
            torch.Tensor: A filtered tensor containing only those masks whose areas are within k standard deviations of the mean area.
        """
        # Calculate the area of each mask by summing the flattened values
        areas = (masks.flatten(1)).sum(dim=-1)

        # Compute the mean and population standard deviation of the areas
        mu = torch.mean(areas)
        sigma = torch.std(areas, unbiased=False)  # Using population std

        # Determine the lower and upper bounds based on k standard deviations
        lb = mu - k * sigma
        ub = mu + k * sigma

        # Create a boolean mask for areas within [lb, ub]
        valid_indices = (areas >= lb) & (areas <= ub)

        # Filter the masks using the valid indices
        filtered_masks = masks[valid_indices]

        return filtered_masks

    def _group_small_masks_by_instance(self, large_masks, small_masks, threshold=0.5):
        """
        Groups small masks based on their overlap with large masks.

        Args:
            large_masks (torch.Tensor): A tensor of shape (N, H, W), where N is the number of large masks.
            small_masks (torch.Tensor): A tensor of shape (M, H, W), where M is the number of small masks.
            threshold (float): A float between 0 and 1, representing the minimum overlap percentage required for a small mask to be considered valid.

        Returns:
            list: A list of N torch.Tensors, where each tensor has shape (K_i, H, W),
                 K_i being the number of small masks that overlap with the i-th large mask.
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

        # Convert lists of masks to tensors
        tensor_grouped_masks = []
        for mask_group in grouped_masks:
            if mask_group:  # If the group is not empty
                # Stack masks along first dimension
                tensor_group = torch.stack(mask_group, dim=0)  # Shape: (K_i, H, W)
                tensor_grouped_masks.append(tensor_group)
            else:
                # Create empty tensor with correct shape if no masks in group
                H, W = large_masks.shape[1:]
                tensor_grouped_masks.append(
                    torch.zeros(
                        (0, H, W), dtype=large_masks.dtype, device=large_masks.device
                    )
                )

        return tensor_grouped_masks

    def cal_closure(self, berry_masks, cluster_masks):
        closures = []
        for berry_mask, cluster_mask in zip(berry_masks, cluster_masks):
            closure = berry_mask.bool().any(dim=0).sum() / cluster_mask.bool().sum()
            closures.append(closure.item())

        return closures

    def analysis_area_distribution_as_figure(self, masks, name):
        areas = masks.sum(dim=(1, 2))
        plt.clf()
        plt.hist(areas.cpu().numpy(), color="skyblue", edgecolor="black")
        plt.xlabel("Area (pixels)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Mask Areas")
        plt.savefig(os.path.join(self.img_save_path, f"{name}_distribution.png"))

    #  a csv file that list image name, cluster closure, berry number.
    def process_folder(self, input_folder):
        """Process all images in the input folder through the grape detection pipeline."""
        csv_results = []
        format = ["png", "jpg", "jpeg"]
        image_files = [f for f in os.listdir(input_folder) if f.endswith(tuple(format))]
        pbar = tqdm(total=len(image_files), desc="Processing images", unit="image")
        process = psutil.Process()

        for i, filename in enumerate(image_files):
            try:
                img_path = os.path.join(input_folder, filename)
                # Use with statement to ensure proper image closure
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    short_name = os.path.splitext(filename)[0]

                    # Process in steps to allow memory cleanup
                    berry_result = self.segment_berry(img)
                    if berry_result[1] is None:
                        continue

                    img, berry_masks_cpu, berry_scores_cpu = berry_result
                    berry_masks_cpu = berry_masks_cpu.squeeze(1)

                    # Immediately release unused variables
                    del berry_scores_cpu

                    grape_instances = self.segment_grape_cluster(img)
                    self.check_memory_limit()
                    if grape_instances is None:
                        print(f"{short_name} instance segmentation result is None.")
                        continue

                    # Filtering and grouping operations
                    # self.analysis_area_distribution_as_figure(
                    #     berry_masks_cpu, short_name + "_before_filter"
                    # )
                    berry_masks_cpu = self.filter_masks_by_iqr(berry_masks_cpu)

                    # self.analysis_area_distribution_as_figure(
                    #     berry_masks_cpu, short_name + "_after_filter"
                    # )

                    filtered_berry_masks = self._group_small_masks_by_instance(
                        grape_instances, berry_masks_cpu, 0.9
                    )

                    # Release berry_masks_cpu before calculating closure
                    del berry_masks_cpu

                    all_filtered_berry_masks = torch.cat(filtered_berry_masks, dim=0)
                    closures = self.cal_closure(filtered_berry_masks, grape_instances)

                    # Save results with explicit grape instance indices
                    csv_results.append(
                        {
                            "image_name": filename,
                            "grape_cluster_num": grape_instances.shape[0],
                            "total_berry_num": all_filtered_berry_masks.shape[0],
                            "closures": {
                                f"grape_cluster_{i}": f"{closure:.2f}"  # More descriptive keys
                                for i, closure in enumerate(closures)
                            },
                            "closure_mean": f"{np.mean(closures):.2f}",
                        }
                    )

                    # Optional: Save CSV every n images
                    if (i + 1) % 10 == 0:
                        df = pd.DataFrame(csv_results)
                        temp_csv_path = os.path.join(
                            self.img_save_path, f"berry_counts_temp.csv"
                        )
                        df.to_csv(temp_csv_path, index=False)

                    # Display and save image
                    show_grape_and_berry(
                        img,
                        grape_instances,
                        all_filtered_berry_masks,
                        title=short_name + "_all_masks",
                        alpha=0.6,
                        save_path=self.img_save_path,
                        dpi=100,
                        show_grape_indices=True,
                    )
                    self.check_memory_limit()

                    # Clean up memory
                    del all_filtered_berry_masks, grape_instances, filtered_berry_masks
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                print(f"Error processing image {filename}: {str(e)}")
                continue

            except MemoryError as e:
                print("MemoryError: Not enough memory available! Reduce memory usage.")
                continue

            # Update progress bar
            if self.device.type == "cuda":
                used_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                total_memory = (
                    torch.cuda.get_device_properties(0).total_memory
                    / 1024
                    / 1024
                    / 1024
                )
            else:
                used_memory = process.memory_info().rss / 1024 / 1024 / 1024
                total_memory = psutil.virtual_memory().total / 1024 / 1024 / 1024

            pbar.set_postfix({"Memory": f"{used_memory:.1f}/{total_memory:.1f}GB)"})
            pbar.update(1)

        pbar.close()

        # Save final results
        df = pd.DataFrame(csv_results)
        random_str = str(random.randint(100, 999))
        csv_path = os.path.join(self.img_save_path, f"berry_counts_{random_str}.csv")
        df.to_csv(csv_path, index=False)


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

    args = parser.parse_args()

    grape_pipeline = GrapePipeline(
        args.point_ckpt, args.mask_ckpt, args.output, sam_from_pretrained=args.sam_pth
    )
    grape_pipeline.process_folder(args.input)


"""
python3 pipeline.py --point-ckpt /home/xz/Dev/GrapeSAM/weights/point/best_val.pth \
--mask-ckpt /home/xz/Dev/GrapeSAM/weights/mask2former/model_0214999.pth \
--input /home/xz/Dev/baseline-exp-playground/DATASET/vivid-close/test \
--output /home/xz/Pictures/vivid-close-output
"""

if __name__ == "__main__":
    main()
