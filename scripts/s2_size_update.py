import json
import os
from PIL import Image


def clean_and_rotate_image(image_path):
    try:
        img = Image.open(image_path)
        # 创建一个没有EXIF数据的新图片
        data = list(img.getdata())
        image_without_exif = Image.new(img.mode, img.size)
        image_without_exif.putdata(data)
        # 顺时针旋转90度
        rotated_image = image_without_exif.rotate(-90, expand=True)
        # 保存处理后的图片
        rotated_image.save(image_path)
        return True
    except Exception as e:
        print(f"处理图片时出错 - {image_path}: {str(e)}")
        return False


def check_image_sizes():
    # 读取 JSON 文件
    json_path = "./merged_coco.json"
    imgs_dir = "/home/xz/Documents/Vivid/imgs"

    # 统计变量
    total_images = 0
    missing_images = 0
    mismatch_images = 0
    error_images = 0
    matched_images = 0

    with open(json_path, "r") as f:
        annotations = json.load(f)

    total_images = len(annotations["images"])

    # 遍历 JSON 中的每个图片记录
    for image_info in annotations["images"]:
        json_width = image_info["width"]
        json_height = image_info["height"]
        image_filename = image_info["file_name"]
        image_path = os.path.join(imgs_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"警告：图片文件不存在 - {image_filename}")
            missing_images += 1
            continue

        try:
            with Image.open(image_path) as img:
                actual_width, actual_height = img.size

                if json_width != actual_width or json_height != actual_height:
                    print(f"尺寸不匹配 - {image_filename}:")
                    print(
                        f" JSON中记录: {json_width}x{json_height}; 实际尺寸: {actual_width}x{actual_height}"
                    )

                    # 处理不匹配的图片
                    print(f"正在处理图片: {image_filename}")
                    if clean_and_rotate_image(image_path):
                        # 重新检查处理后的图片尺寸
                        with Image.open(image_path) as processed_img:
                            new_width, new_height = processed_img.size
                            print(f" 处理后尺寸: {new_width}x{new_height}")
                            if new_width == json_width and new_height == json_height:
                                print(f" 处理成功 - 尺寸现在匹配")
                                matched_images += 1
                            else:
                                print(f" 处理后仍不匹配")
                                mismatch_images += 1
                    else:
                        mismatch_images += 1
                else:
                    matched_images += 1
        except Exception as e:
            print(f"读取图片出错 - {image_filename}: {str(e)}")
            error_images += 1

    # 打印统计结果
    print("\n统计结果:")
    print(f"总图片数量: {total_images}")
    print(f"尺寸匹配数量: {matched_images}")
    print(f"尺寸不匹配数量: {mismatch_images}")
    print(f"缺失图片数量: {missing_images}")
    print(f"读取错误数量: {error_images}")


if __name__ == "__main__":
    check_image_sizes()
