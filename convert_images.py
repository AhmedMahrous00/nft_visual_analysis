import os
import pathlib
import cv2
import imageio
import numpy as np
from tqdm import tqdm
import argparse

TARGET_FORMAT = ".png"

# Common image extensions to look for (case-insensitive)
VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
# Simple extension corrections
EXTENSION_CORRECTIONS = {
    '.gifo': '.gif',
    '.jpe': '.jpeg',
    # Add more if needed
}

def ensure_dir(file_path):
    """Ensure the directory for the given file_path exists."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def correct_extension(filename):
    """Corrects known problematic extensions."""
    name, ext = os.path.splitext(filename)
    ext_lower = ext.lower()

    if ext_lower in EXTENSION_CORRECTIONS:
        new_ext = EXTENSION_CORRECTIONS[ext_lower]
        return name + new_ext, new_ext
    
    for valid_ext in VALID_IMAGE_EXTENSIONS:
        if ext_lower == valid_ext:
            if ext != valid_ext: 
                 return name + valid_ext, valid_ext
            return filename, ext_lower 
    return filename, ext_lower


def process_images(source_dir, target_dir):
    """Scans the source directory, converts images to PNG, and saves them to the target directory."""
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return

    if not os.path.exists(target_dir):
        print(f"Creating target directory: {target_dir}")
        os.makedirs(target_dir)

    processed_count = 0
    error_count = 0
    error_files = []

    files_to_process = []
    for root, _, files in os.walk(source_dir):
        for filename in files:
            corrected_filename, ext_lower = correct_extension(filename)
            if ext_lower in VALID_IMAGE_EXTENSIONS or ext_lower in EXTENSION_CORRECTIONS.values():
                files_to_process.append(pathlib.Path(root) / corrected_filename)

    for source_filepath in tqdm(files_to_process, desc="Processing images"):
        try:
            relative_path = source_filepath.relative_to(source_dir)
            target_filepath_no_ext = pathlib.Path(target_dir) / relative_path.parent / source_filepath.stem
            target_filepath_png = target_filepath_no_ext.with_suffix(TARGET_FORMAT)

            ensure_dir(target_filepath_png)

            img = None
            img_cv = cv2.imread(str(source_filepath))
            if img_cv is not None:
                img = img_cv
            else:
                img_pil = imageio.v2.imread(source_filepath)
                img_np = np.array(img_pil)

                if img_np.ndim == 2:
                    img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                elif img_np.shape[2] == 4:
                    img = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                elif img_np.shape[2] == 3:
                    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                else:
                    error_count +=1
                    error_files.append(str(source_filepath))
                    continue
            
            if img is None:
                error_count += 1
                error_files.append(str(source_filepath))
                continue

            cv2.imwrite(str(target_filepath_png), img)
            processed_count += 1

        except Exception as e:
            error_count += 1
            error_files.append(str(source_filepath))
    
    print(f"\n--- Conversion Summary ---")
    print(f"Successfully processed: {processed_count} images.")
    print(f"Failed to process: {error_count} images.")

    if error_files:
        error_log_path = pathlib.Path(target_dir) / "conversion_errors.log"
        print(f"Writing list of {len(error_files)} problematic files to: {error_log_path}")
        with open(error_log_path, 'w') as f:
            for filepath in error_files:
                f.write(f"{filepath}\n")
    else:
        print("No errors encountered during conversion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images in a directory to PNG format.")
    parser.add_argument('--source_dir', type=str, required=True, help='Directory containing original images')
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to save converted PNG images')
    args = parser.parse_args()
    process_images(args.source_dir, args.target_dir) 