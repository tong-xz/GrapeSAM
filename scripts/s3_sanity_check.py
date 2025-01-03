import json
import os
from PIL import Image
import colorama
from colorama import Fore, Style

# Global path variables
JSON_PATH = "/home/xz/Documents/Vivid/ann.json"
IMGS_DIR = "/home/xz/Documents/Vivid/imgs"
NPY_DIR = "/home/xz/Documents/Vivid/anns"  # 新增 NPY 文件夹路径
EXPECTED_IMG_COUNT = 5000

# 初始化 colorama
colorama.init()


def log_success(message):
    print(f"{Fore.GREEN}[通过] {message}{Style.RESET_ALL}")


def log_error(message):
    print(f"{Fore.RED}[失败] {message}{Style.RESET_ALL}")


def log_info(message):
    print(f"[信息] {message}")


def check_image_sizes():
    # 统计变量
    total_images = 0
    missing_images = 0
    mismatch_images = 0
    error_images = 0
    matched_images = 0

    print("开始数据集完整性检查...\n")

    # Case 1: 检查文件夹中的实际图片数量
    actual_img_files = set(os.listdir(IMGS_DIR))
    actual_npy_files = set(os.listdir(NPY_DIR))  # 新增 NPY 文件检查

    print("\n检查项 1: 文件数量验证")
    if len(actual_img_files) != EXPECTED_IMG_COUNT:
        log_error(
            f"图片文件夹中的图片数量 ({len(actual_img_files)}) 与预期数量 ({EXPECTED_IMG_COUNT}) 不符"
        )
    else:
        log_success(f"图片文件夹包含预期数量的图片: {EXPECTED_IMG_COUNT}")

    if len(actual_npy_files) != EXPECTED_IMG_COUNT:
        log_error(
            f"NPY文件夹中的文件数量 ({len(actual_npy_files)}) 与预期数量 ({EXPECTED_IMG_COUNT}) 不符"
        )
    else:
        log_success(f"NPY文件夹包含预期数量的文件: {EXPECTED_IMG_COUNT}")

    # Case 2: 检查JSON文件中的图片数量
    with open(JSON_PATH, "r") as f:
        annotations = json.load(f)

    total_images = len(annotations["images"])
    print("\n检查项 2: JSON文件图片数量验证")
    if total_images != EXPECTED_IMG_COUNT:
        log_error(
            f"JSON文件中的图片数量 ({total_images}) 与预期数量 ({EXPECTED_IMG_COUNT}) 不符"
        )
    else:
        log_success(f"JSON文件包含预期数量的图片记录: {EXPECTED_IMG_COUNT}")

    # Case 3: 检查文件名对齐
    print("\n检查项 3: 文件名对齐验证")
    json_img_names = set(img["file_name"] for img in annotations["images"])
    npy_names = {
        f"{os.path.splitext(f)[0]}.npy" for f in json_img_names
    }  # 生成期望的 NPY 文件名

    # 3.1: 检查JSON中的文件是否都存在于文件夹中
    missing_in_folder = json_img_names - actual_img_files
    missing_npy = npy_names - actual_npy_files

    if missing_in_folder:
        log_error(f"发现{len(missing_in_folder)}个JSON中的图片在文件夹中不存在")
        for img_name in missing_in_folder:
            print(f"  - {img_name}")
    else:
        log_success("JSON中的所有图片都存在于文件夹中")

    if missing_npy:
        log_error(f"发现{len(missing_npy)}个NPY文件缺失")
        for npy_name in missing_npy:
            print(f"  - {npy_name}")
    else:
        log_success("所有需要的NPY文件都存在")

    # 3.2: 检查文件夹中的文件是否都存在于JSON中
    extra_in_folder = actual_img_files - json_img_names
    extra_npy = actual_npy_files - npy_names

    if extra_in_folder:
        log_error(f"发现{len(extra_in_folder)}个文件夹中的图片在JSON中不存在")
        for img_name in extra_in_folder:
            print(f"  - {img_name}")
    else:
        log_success("文件夹中的所有图片都在JSON中有记录")

    if extra_npy:
        log_error(f"发现{len(extra_npy)}个多余的NPY文件")
        for npy_name in extra_npy:
            print(f"  - {npy_name}")
    else:
        log_success("NPY文件夹中没有多余文件")

    # 遍历 JSON 中的每个图片记录
    for image_info in annotations["images"]:
        json_width = image_info["width"]
        json_height = image_info["height"]
        image_filename = image_info["file_name"]

        # 获取实际图片路径
        image_path = os.path.join(IMGS_DIR, image_filename)

        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            print(f"警告：图片文件不存在 - {image_filename}")
            missing_images += 1
            continue

        # 读取实际图片尺寸
        try:
            with Image.open(image_path) as img:
                actual_width, actual_height = img.size

                # 比较尺寸
                if json_width != actual_width or json_height != actual_height:
                    print(f"尺寸不匹配 - {image_filename}:")
                    print(
                        f" {mismatch_images} JSON中记录: {json_width}x{json_height}; 实际尺寸: {actual_width}x{actual_height}"
                    )
                    mismatch_images += 1
                else:
                    matched_images += 1
        except Exception as e:
            print(f"读取图片出错 - {image_filename}: {str(e)}")
            error_images += 1

    # 更新最终统计结果的格式
    print("\n检查项 4: 图片尺寸验证统计")
    log_info(f"总图片数量: {total_images}")
    log_info(f"尺寸匹配数量: {matched_images}")
    log_info(f"尺寸不匹配数量: {mismatch_images}")
    log_info(f"缺失图片数量: {missing_images}")
    log_info(f"读取错误数量: {error_images}")


if __name__ == "__main__":
    check_image_sizes()
