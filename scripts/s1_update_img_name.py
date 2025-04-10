import os


def convert_extensions_to_lowercase(directory):
    # Get all files in the directory
    for filename in os.listdir(directory):
        # Get the file path
        file_path = os.path.join(directory, filename)

        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            # Split filename into name and extension
            name, ext = os.path.splitext(filename)

            # If the extension has uppercase letters
            if ext.lower() != ext:
                # Create new filename with lowercase extension
                new_filename = name + ext.lower()
                new_file_path = os.path.join(directory, new_filename)

                # Rename the file
                os.rename(file_path, new_file_path)
                print(f"Renamed: {filename} → {new_filename}")


if __name__ == "__main__":
    # Specify your directory path
    directory = "/data/datasets/grape/Vivid/images/"

    # Run the conversion
    convert_extensions_to_lowercase(directory)
    print("Conversion complete!")
