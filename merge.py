import os
from coco_assistant import COCO_Assistant

from pycocotools.coco import COCO
import json


def find_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                json_files.append(full_path)
    return json_files


def merge_coco_json(json_files, output_file):
    merged_annotations = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    image_id_offset = 0
    annotation_id_offset = 0
    category_id_offset = 0
    existing_category_ids = set()

    for idx, file in enumerate(json_files):
        coco = COCO(file)

        # Update image IDs to avoid conflicts
        for image in coco.dataset["images"]:
            image["id"] += image_id_offset
            merged_annotations["images"].append(image)

        # Update annotation IDs to avoid conflicts
        for annotation in coco.dataset["annotations"]:
            annotation["id"] += annotation_id_offset
            annotation["image_id"] += image_id_offset
            merged_annotations["annotations"].append(annotation)

        # Update categories and their IDs to avoid conflicts
        for category in coco.dataset["categories"]:
            if category["id"] not in existing_category_ids:
                category["id"] += category_id_offset
                merged_annotations["categories"].append(category)
                existing_category_ids.add(category["id"])

        image_id_offset = len(merged_annotations["images"])
        annotation_id_offset = len(merged_annotations["annotations"])
        category_id_offset = len(merged_annotations["categories"])

    # Save merged annotations to output file
    with open(output_file, "w") as f:
        json.dump(merged_annotations, f)


if __name__ == "__main__":
    directory = "/home/xz/Documents/Vivid/mask-anns"
    json_files = find_json_files(directory)

    # Output file path for merged annotations
    output_file = "merged_coco.json"

    # Merge COCO JSON files
    merge_coco_json(json_files, output_file)

    print("Merged COCO JSON files saved to", output_file)
