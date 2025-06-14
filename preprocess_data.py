import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_mean_rgb_kitti(dataset_path):
    """
    Calculate mean RGB values across all sequence folders in the KITTI dataset, looking into `image_3` subfolders.
    """
    sequence_folders = [os.path.join(dataset_path, seq, "image_3") for seq in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, seq, "image_3"))]
    total_pixels = 0
    sum_rgb = np.zeros(3, dtype=np.float64)

    for seq_folder in sequence_folders:
        image_files = [os.path.join(seq_folder, f) for f in os.listdir(seq_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

        for file in tqdm(image_files, desc=f"Calculating Mean RGB in {seq_folder}"):
            image = Image.open(file).convert("RGB")
            image_array = np.array(image, dtype=np.float64)
            sum_rgb += image_array.sum(axis=(0, 1))
            total_pixels += image_array.shape[0] * image_array.shape[1]

    mean_rgb = sum_rgb / total_pixels
    return mean_rgb

def preprocess_and_save_kitti(dataset_path, output_path, mean_rgb):
    """
    Preprocess KITTI dataset images by subtracting mean RGB values and save them to an output directory.
    """
    sequence_folders = [os.path.join(dataset_path, seq, "image_3") for seq in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, seq, "image_3"))]

    for seq_folder in sequence_folders:
        sequence_name = os.path.basename(os.path.dirname(seq_folder))  # Extract sequence folder name (e.g., `00`)
        output_seq_folder = os.path.join(output_path, sequence_name, "image_3")
        os.makedirs(output_seq_folder, exist_ok=True)

        image_files = [os.path.join(seq_folder, f) for f in os.listdir(seq_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

        for file in tqdm(image_files, desc=f"Processing Images in {seq_folder}"):
            image = Image.open(file).convert("RGB")
            image_array = np.array(image, dtype=np.float32)

            # Subtract mean RGB values
            preprocessed_array = image_array - mean_rgb

            # Clip values to valid range [0, 255] and convert to uint8
            preprocessed_array = np.clip(preprocessed_array, 0, 255).astype(np.uint8)

            # Convert back to an image
            preprocessed_image = Image.fromarray(preprocessed_array)

            # Save the processed image
            output_file = os.path.join(output_seq_folder, os.path.basename(file))
            preprocessed_image.save(output_file)

# Paths
kitti_dataset_path = "dataset/sequences"  # Replace with the path to the KITTI dataset
kitti_output_path = "processed_dataset/sequences"  # Replace with the path to save processed images

# Step 1: Calculate mean RGB across all sequences
mean_rgb = (0.19007764876619865, 0.15170388157131237, 0.10659445665650864)
print("Calculated Mean RGB:", mean_rgb)

# Step 2: Preprocess and save all images
preprocess_and_save_kitti(kitti_dataset_path, kitti_output_path, mean_rgb)
