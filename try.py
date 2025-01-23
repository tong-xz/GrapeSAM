import os
import random
from collections import Counter


def get_files_in_directory():
    directory = "/home/xz/Documents/Vivid/imgs"
    files = os.listdir(directory)
    return files


def read_filenames_from_txt(txt_path):
    """
    从文本文件中读取文件名列表
    Args:
        txt_path: 文本文件的路径
    Returns:
        包含所有文件名的列表
    """
    filenames = []
    try:
        with open(txt_path, "r") as file:
            filenames = [
                line.strip() for line in file if line.strip()
            ]  # Only include non-empty lines
        return filenames
    except FileNotFoundError:
        print(f"错误：找不到文件 {txt_path}")
        return []
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
        return []


train_txt_filenames = read_filenames_from_txt("/home/xz/Documents/Vivid/train.txt")
val_txt_filenames = read_filenames_from_txt("/home/xz/Documents/Vivid/val.txt")
test_txt_filenames = read_filenames_from_txt("/home/xz/Documents/Vivid/test.txt")
txt_files = train_txt_filenames + val_txt_filenames + test_txt_filenames
img_files = get_files_in_directory()

# Print the counts of train, validation and test files
print(f"Number of training files: {len(train_txt_filenames)}")
print(f"Number of validation files: {len(val_txt_filenames)}")
print(f"Number of test files: {len(test_txt_filenames)}")

# Find image files not listed in any of the text files
img_files_set = set(img_files)
txt_files_set = set(txt_files)
img_not_in_txt = img_files_set - txt_files_set

# Print the image files not present in the text files
print("Image files not in text files:")
for img in img_not_in_txt:
    print(img)
