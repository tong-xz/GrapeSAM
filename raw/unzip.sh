#!/bin/bash

# Function to prompt for yes/no confirmation
confirm() {
    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Create output directories
mkdir -p images anns

# Counter for extracted files
extracted_count=0

# Loop through all zip files in the current directory
for zipfile in *.zip
do
    # Check if the file exists (to handle the case when no zip files are found)
    if [ -f "$zipfile" ]; then
        # Get the filename without extension
        filename=$(basename "$zipfile" .zip)
        
        # Create a temporary directory for extraction
        temp_dir="temp_$filename"
        mkdir -p "$temp_dir"
        
        # Unzip the file into the temporary directory
        unzip -q "$zipfile" -d "$temp_dir"
        
        # Find and copy all image files from the specific path
        if [ -d "$temp_dir/images/default/redoexamples" ]; then
            find "$temp_dir/images/default/redoexamples" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.bmp" -o -iname "*.tiff" \) -exec cp -p {} images/ \;
        else
            echo "Warning: images/default/redoexamples not found in $zipfile"
        fi
        
        # Find and copy all JSON files to the anns directory
        find "$temp_dir" -type f -name "*.json" -exec sh -c 'cp -p "$0" "anns/'"$filename"'_$(basename "$0")"' {} \;
        
        # Remove the temporary directory
        rm -rf "$temp_dir"
        
        echo "Processed $zipfile"
        
        ((extracted_count++))
    fi
done

echo "All zip files have been processed. Total files processed: $extracted_count"
echo "Images copied to 'images' directory: $(find images -type f | wc -l)"
echo "JSON files copied to 'anns' directory: $(find anns -type f | wc -l)"

# Ask user if they want to delete the original zip files
if [ $extracted_count -gt 0 ] && confirm "Do you want to delete the original ZIP files?"; then
    rm *.zip
    echo "Original ZIP files have been deleted."
else
    echo "Original ZIP files have been kept."
fi


