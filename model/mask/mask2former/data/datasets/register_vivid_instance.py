import random
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import os


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
import os


def filter_images_without_masks(dataset_dicts):
    """
    Filters out images that do not have valid ground-truth masks (segmentation).
    """
    filtered_dataset = []
    excluded_files = []
    size_mismatch_files = ["IMG_0119.png"]
    for data in dataset_dicts:
        annotations = data.get("annotations", [])
        # Check if any annotation has a valid segmentation key
        if data["file_name"].split("/")[-1] in size_mismatch_files:
            excluded_files.append(data["file_name"])
            continue

        if all("segmentation" in ann and ann["segmentation"] for ann in annotations):
            filtered_dataset.append(data)
        else:
            excluded_files.append(data["file_name"])
    print(f"Excluded images: {excluded_files}")
    return filtered_dataset


def split_dataset(dataset_dicts, val_ratio=0.2):
    """
    Split a dataset dictionary into training and validation sets.

    Args:
        dataset_dicts (list): List of dataset dictionaries.
        val_ratio (float): Proportion of the dataset to use for validation.

    Returns:
        tuple: Two lists (train_dicts, val_dicts)
    """
    random.seed(42)
    random.shuffle(dataset_dicts)
    split_idx = int(len(dataset_dicts) * (1 - val_ratio))
    train_dicts = dataset_dicts[:split_idx]
    val_dicts = dataset_dicts[split_idx:]
    return train_dicts, val_dicts


def register_vivid_datasets():
    # Define dataset paths
    dataset_root = "./Vivid" # [TODO] you need to change this path to your own dataset location
    images_path = os.path.join(dataset_root, "imgs")
    # images_path = os.path.join(dataset_root)
    annotations_path = os.path.join(dataset_root, "anns/instances_default_v4.json")

    # Load dataset from the combined annotation file
    dataset_dicts = load_coco_json(annotations_path, images_path, "vivid_default")
    # breakpoint()
    # dataset_dicts = make_name(dataset_dicts)
    dataset_dicts = filter_images_without_masks(dataset_dicts)

    # Split the dataset into training and validation sets
    train_dicts, val_dicts = split_dataset(dataset_dicts, val_ratio=0.2)

    # Register filtered datasets
    DatasetCatalog.register("vivid_train", lambda: train_dicts)
    MetadataCatalog.get("vivid_train").set(
        thing_classes=["grape"], evaluator_type="coco"
    )

    DatasetCatalog.register("vivid_val", lambda: val_dicts)
    MetadataCatalog.get("vivid_val").set(thing_classes=["grape"], evaluator_type="coco")

    print("VIVID datasets registered with filtered annotations!")


# if you need to train the model, uncomment the following line
register_vivid_datasets()

if __name__ == "__main__":

    train_dataset = DatasetCatalog.get("vivid_train")
    val_dataset = DatasetCatalog.get("vivid_val")
    breakpoint()  # need to find the number issues.

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print("VIVID datasets registered with filtered annotations!")

    # verify that all annotations have segmentation fields
    for d in train_dataset:
        for obj in d["annotations"]:
            if "segmentation" not in obj:
                print(f'{d["file_name"]} has an annotation with no segmentation field')

    # verify that all image_id do not have duplicates
    image_ids = [d["image_id"] for d in train_dataset]
    image_ids.extend([d["image_id"] for d in val_dataset])
    assert len(image_ids) == len(set(image_ids)), "Image IDs have duplicates!"
    print(f"All {len(image_ids)} image IDs are unique!")
