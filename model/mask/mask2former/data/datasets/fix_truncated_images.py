from register_vivid_instance import register_vivid_datasets # do not need, but need to import
from detectron2.data import DatasetCatalog
from PIL import Image, UnidentifiedImageError
import numpy as np

def find_truncated_images(dataset_name):
    """
    Finds truncated or corrupted images in a Detectron2 dataset.

    Args:
        dataset_name (str): Name of the registered dataset in DatasetCatalog.

    Returns:
        list: List of file paths of truncated/corrupted images.
    """
    dataset = DatasetCatalog.get(dataset_name)
    corrupted_images = []

    for item in dataset:
        image_path = item[
            "file_name"
        ]  # Detectron2 uses "file_name" for image paths
        try:
            # Attempt to open the image
            img = Image.open(image_path)
            # img.verify()  # Verify image integrity
            img_np = np.array(img)

            # img.close()
        except (OSError, UnidentifiedImageError, IOError) as e:
            print(f"Corrupted image detected: {image_path}")
            corrupted_images.append(image_path)

    return corrupted_images

# Detect truncated images in training and validation datasets
train_corrupted_images = find_truncated_images("vivid_train")
val_corrupted_images = find_truncated_images("vivid_val")

print("Corrupted images in train dataset:", train_corrupted_images)
print("Corrupted images in val dataset:", val_corrupted_images)

# just need to save the image again.