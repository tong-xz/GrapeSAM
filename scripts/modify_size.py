from PIL import Image


def clean_exif_data(image_path):
    try:
        # 打开图片
        img = Image.open(image_path)
        print(f"原图尺寸: {img.size}")

        # 创建一个没有EXIF数据的新图片
        data = list(img.getdata())
        image_without_exif = Image.new(img.mode, img.size)
        image_without_exif.putdata(data)

        # 顺时针旋转90度
        rotated_image = image_without_exif.rotate(
            -90, expand=True
        )  # 负数表示顺时针旋转

        # 保存清理并旋转后的图片 - 直接覆盖原文件
        rotated_image.save(image_path)
        print(f"已清理EXIF数据并旋转90度后保存到: {image_path}")

        # 使用多种方法验证处理后的图片尺寸
        # 1. 使用 PIL
        processed_img = Image.open(image_path)  # 读取更新后的原文件
        pil_size = processed_img.size

        # 2. 使用 cv2
        import cv2

        cv_img = cv2.imread(image_path)
        cv_size = (
            cv_img.shape[1],
            cv_img.shape[0],
        )  # cv2的尺寸是(height, width)，需要转换

        # 3. 获取文件属性
        import os

        file_size = os.path.getsize(image_path)

        print("\n尺寸验证结果:")
        print(f"PIL方式读取尺寸: {pil_size}")
        print(f"OpenCV方式读取尺寸: {cv_size}")
        print(f"文件大小: {file_size/1024:.2f} KB")

        # 验证尺寸是否一致
        if pil_size != cv_size:
            print("警告：不同方法读取的图片尺寸不一致！")

    except Exception as e:
        print(f"处理图片时出错: {e}")


# 清理图片的EXIF数据
image_path = "/home/xz/Documents/1597923993611.jpg"
clean_exif_data(image_path)
