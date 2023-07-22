import sys
from py_helpers.general_helpers import *
import numpy as np

import SimpleITK as sitk

import shutil

import subprocess
import nibabel as nib

# Flip the image
def flip_image(image):

    # Get the image array
    image = sitk.ReadImage(image)
    image_array = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))

    # Get the spacing, direction and origin of the image
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    # Flip the image array
    image_array = np.flipud(image_array)

    # Get the new image
    new_image = sitk.GetImageFromArray(np.transpose(image_array, (2, 1, 0)))

    # Set the new spacing, direction and origin
    new_image.SetSpacing(spacing)
    new_image.SetDirection(direction)
    new_image.SetOrigin(origin)

    # Return the new image
    return new_image


# Main function
def main():
    # Define the path to the data
    data_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/processed_dMRI/dMRI_unzipped/"

    # Grab all the nii.gz files
    nii_gz_files = glob_files(data_path, "nii.gz")

    # Filter for the b0 images
    b0_images = [file for file in nii_gz_files if "b0" in file]

    # Print the length of the b0 images
    print(len(b0_images))

    # Define a new path
    data_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/processed_dMRI/dMRI_b0"
    check_output_folders(data_path, "b0", wipe=False)

    for b0 in b0_images:

        # Get the region ID
        region_ID = b0.split(os.sep)[-3]

        # Create new fodler
        new_folder = os.path.join(data_path, region_ID)
        check_output_folders(new_folder, "b0", wipe=False)

        # Get the filename
        filename = b0.split(os.sep)[-1]

        # Define the new path
        new_path = os.path.join(new_folder, filename)

        # Copy the file
        print("Copying {} to {}".format(b0, new_path))
        shutil.copyfile(b0, new_path)




if __name__ == "__main__":
    main()
