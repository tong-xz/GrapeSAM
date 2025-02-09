import os
from pillow_heif import register_heif_opener
from PIL import Image
import random
from tqdm import tqdm


def convert_heic_imgs(input_path, output_path, output_format="png"):
    """
    Convert a single HEIC image to the specified format.

    Args:
        input_path (str): Path to the input HEIC image
        output_path (str): Path to save the converted image
        output_format (str): Output format (e.g., 'png', 'jpeg', 'jpg')

    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Register HEIF opener to handle HEIC files
        register_heif_opener()

        # Open and convert the image
        with Image.open(input_path) as img:
            # Convert to RGB mode if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Save with the specified format
            img.save(output_path, output_format)
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False


def process_folder(input_folder, output_folder, output_format="png"):
    """
    Process all HEIC/HEIF images in a folder and convert them to the specified format.

    Args:
        input_folder (str): Path to folder containing HEIC images
        output_folder (str): Path to output folder
        output_format (str): Output format (e.g., 'png', 'jpeg', 'jpg')

    Returns:
        tuple: (int, int) Number of successful conversions and total files processed
    """
    successful = 0
    total = 0

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get list of HEIC files first
    heic_files = [
        f for f in os.listdir(input_folder) if f.lower().endswith((".heic", ".heif"))
    ]
    total = len(heic_files)

    # Process all HEIC files with progress bar
    for filename in tqdm(heic_files, desc="Converting images", unit="image"):
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + "." + output_format
        output_path = os.path.join(output_folder, output_filename)

        # Check if file already exists and add random number if it does
        while os.path.exists(output_path):
            random_num = (
                f"{random.randint(0, 999):03d}"  # Generate 3-digit random number
            )
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_{random_num}.{output_format}"
            output_path = os.path.join(output_folder, output_filename)

        if convert_heic_imgs(input_path, output_path, output_format):
            successful += 1

    return successful, total


def main():
    """
    Main function to handle command line arguments and process HEIC images.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HEIC images to other formats."
    )
    parser.add_argument(
        "--input-folder", help="Path to original folder containing HEIC images"
    )
    parser.add_argument(
        "--output-folder",
        help="Path to output folder (default: input_folder/converted)",
        default=None,
    )
    parser.add_argument(
        "--format",
        help="Output format (default: png)",
        default="png",
        choices=["png", "jpg", "jpeg"],
    )

    args = parser.parse_args()

    # If output folder not specified, create 'converted' subfolder in input folder
    if args.output_folder is None:
        args.output_folder = os.path.join(args.input_folder, "converted")

    successful, total = process_folder(
        args.input_folder, args.output_folder, args.format
    )
    print(f"\nCOMPLETE: {successful} of {total} files converted successfully")


if __name__ == "__main__":
    main()
