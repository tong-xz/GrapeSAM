from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
import os


def filter_images_without_masks(dataset_dicts):
    """
    Filters out images that do not have valid ground-truth masks (segmentation).
    """
    filtered_dataset = []
    excluded_files = []
    for data in dataset_dicts:
        annotations = data.get("annotations", [])
        # Check if any annotation has a valid segmentation key
        if all("segmentation" in ann and ann["segmentation"] for ann in annotations):
            filtered_dataset.append(data)
        else:
            excluded_files.append(data["file_name"])
    print(f"Excluded images: {excluded_files}")
    return filtered_dataset


def register_vivid_datasets():
    # Define dataset paths
    dataset_root = "datasets/vivid"
    train_images = os.path.join(dataset_root, "images")
    train_annotations = os.path.join(dataset_root, "annotations/instances_train.json")
    val_images = os.path.join(dataset_root, "images")
    val_annotations = os.path.join(dataset_root, "annotations/instances_val.json")

    # Load and filter datasets
    train_dataset_dicts = load_coco_json(train_annotations, train_images, "vivid_train")
    val_dataset_dicts = load_coco_json(val_annotations, val_images, "vivid_val")

    train_dataset_filtered = filter_images_without_masks(train_dataset_dicts)
    val_dataset_filtered = filter_images_without_masks(val_dataset_dicts)

    # Register filtered datasets
    DatasetCatalog.register("vivid_train", lambda: train_dataset_filtered)
    MetadataCatalog.get("vivid_train").set(
        thing_classes=["grape"], evaluator_type="coco"
    )

    DatasetCatalog.register("vivid_val", lambda: val_dataset_filtered)
    MetadataCatalog.get("vivid_val").set(thing_classes=["grape"], evaluator_type="coco")
    print("VIVID datasets registered with filtered annotations!")


# register_vivid_datasets()

if __name__ == "__main__":
    # register_vivid_datasets()
    train_dataset = DatasetCatalog.get("vivid_train")
    val_dataset = DatasetCatalog.get("vivid_val")

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print("VIVID datasets registered with filtered annotations!")

    for d in train_dataset:
        for obj in d["annotations"]:
            if "segmentation" not in obj:
                print(f'{d["file_name"]} has an annotation with no segmentation field')
