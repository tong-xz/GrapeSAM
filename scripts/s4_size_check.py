import json
from PIL import Image
import os

# 读取JSON文件
file_path = "/home/xz/Documents/Vivid/ann_v3.json"
with open(file_path, "r") as file:
    data = json.load(file)

# 提取图像信息和标注信息
images = {img["id"]: (img["width"], img["height"]) for img in data["images"]}
annotations = data["annotations"]

# 添加图片文件夹路径
img_folder = "/home/xz/Documents/Vivid/imgs"

# 修改验证逻辑
mismatches = []

for annotation in annotations:
    image_id = annotation["image_id"]
    if image_id in images:
        # 从JSON中获取的图像尺寸
        json_size = images[image_id]

        # 从segmentation中获取的尺寸
        mask_size = (
            (
                annotation["segmentation"].get("size")[1],
                annotation["segmentation"].get("size")[0],
            )
            if "size" in annotation["segmentation"]
            else None
        )

        # 获取实际图片文件的尺寸
        img_filename = next(
            (img["file_name"] for img in data["images"] if img["id"] == image_id), None
        )
        if img_filename:
            img_path = os.path.join(img_folder, img_filename)
            try:
                with Image.open(img_path) as img:
                    actual_size = img.size  # (width, height)

                    # 比较三种尺寸
                    if mask_size and (
                        mask_size != json_size
                        or mask_size != actual_size
                        or json_size != actual_size
                    ):
                        mismatches.append(
                            {
                                "file_name": img_filename,
                                "image_id": image_id,
                                "json_size": json_size,
                                "mask_size": mask_size,
                                "actual_size": actual_size,
                            }
                        )
            except Exception as e:
                print(f"无法读取图片 {img_filename}: {str(e)}")

# 修改输出格式
if mismatches:
    print("发现尺寸不匹配的情况：")
    for mismatch in mismatches:
        print(f"\n文件名: {mismatch['file_name']}")
        print(f"Image ID: {mismatch['image_id']}")
        print(f"JSON中的尺寸: {mismatch['json_size']}")
        print(f"Segmentation尺寸: {mismatch['mask_size']}")
        print(f"实际图片尺寸: {mismatch['actual_size']}")
else:
    print("所有尺寸均匹配。")
