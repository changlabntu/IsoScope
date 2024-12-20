import os
import re
from PIL import Image
import numpy as np
import tifffile


def convert_2d_to_3d_tiffs(source_dir, destination_dir):
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Dictionary to store slices for each subject
    subject_slices = {}

    # Regular expression to match the filename pattern
    pattern = r'(\d+_\d+)_(\d+)\.tif'

    # Collect all 2D TIFF files
    for filename in os.listdir(source_dir):
        if filename.endswith('.tif'):
            match = re.match(pattern, filename)
            if match:
                subject_name, slice_number = match.groups()
                slice_number = int(slice_number)

                if subject_name not in subject_slices:
                    subject_slices[subject_name] = []

                # Read the 2D TIFF file
                img_path = os.path.join(source_dir, filename)
                img = Image.open(img_path)
                subject_slices[subject_name].append((slice_number, np.array(img)))

    # Convert and save 3D TIFF files
    for subject_name, slices in subject_slices.items():
        # Sort slices by slice number
        slices.sort(key=lambda x: x[0])

        # Stack the sorted slices into a 3D array
        volume = np.stack([slice_data for _, slice_data in slices], axis=0)

        # Save the 3D TIFF file
        output_filename = f"{subject_name}.tif"
        output_path = os.path.join(destination_dir, output_filename)
        tifffile.imwrite(output_path, volume)

        print(f"Saved 3D TIFF for subject {subject_name}")

    print("Conversion completed.")


#for i in ['dualE', 'dualE-SPADE', 'vanilla', '3D']:
source_dir = '/media/ExtHDD01/oai_diffusion_local/aAPain0908/results/test/0/0/Out/'
destination_dir = '/media/ExtHDD01/oai_diffusion_local/aAPain0908/results/test/0/0/0/'
os.makedirs(destination_dir, exist_ok=True)

convert_2d_to_3d_tiffs(source_dir, destination_dir)