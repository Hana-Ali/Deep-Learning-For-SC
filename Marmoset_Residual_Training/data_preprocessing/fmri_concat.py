import sys
import os
from py_helpers.general_helpers import *
import nibabel as nib
import numpy as np
import subprocess
import glob

# Function to concatenate the files
def concat_files(data_path, nii_gz_files):
    """Concatenate NIfTI files using mrcat from MRtrix3."""

    # Define the starting index
    starting_index = 0

    # Grab all the registered files
    registered_files = glob.glob(os.path.join(data_path, "concat_registered_fmri_*.nii.gz"))

    print("Found {} registered files".format(len(registered_files)))

    if registered_files:
        # Extract the index of the last concatenated file
        starting_index = max([int(f.split("_")[-1].split(".")[0]) for f in registered_files]) + 1

    print("Starting from index {}".format(starting_index))

    # If starting from the beginning
    if starting_index == 0:
        registered_fmri_path = os.path.join(data_path, f"concat_registered_fmri_{starting_index}.nii.gz")
        subprocess.run(['cp', nii_gz_files[0], registered_fmri_path])
        starting_index += 1

    # Now, for each subsequent file, concatenate it with the output file
    for i in range(starting_index, len(nii_gz_files)):
        print("-"*50)
        print("Concatenating file {} of {}".format(i, len(nii_gz_files)))
        registered_fmri_path = os.path.join(data_path, f"concat_registered_fmri_{i}.nii.gz")
        temp_output = f'temp_concat_{i}.nii.gz'
        subprocess.run(['mrcat', os.path.join(data_path, f"concat_registered_fmri_{i-1}.nii.gz"), nii_gz_files[i], temp_output, '-force'])
        
        # Move the temporary output file to the main output file
        subprocess.run(['mv', temp_output, registered_fmri_path])

        # Remove the older file to save space
        if i > 0:
            os.remove(os.path.join(data_path, f"concat_registered_fmri_{i-1}.nii.gz"))

# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    labs = int(sys.argv[2])
    if hpc:
        data_path = "/rds/general/user/hsa22/ephemeral/MBM_fmri/sub-NIHm32/ses-01/fmri_slices/sub-NIHm32_ses-01_task-rest_run-LR-2"
    elif labs:
        data_path = "/media/hsa22/Expansion/MBM_fmri/sub-NIHm32/ses-01/fmri_slices/sub-NIHm32_ses-01_task-rest_run-LR-2"
    else:
        data_path = "/mnt/d/MBM_fmri/sub-NIHm32/ses-01/fmri_slices/sub-NIHm32_ses-01_task-rest_run-LR-2"

    # Define the folder with the registered slices
    registration_folder = os.path.join(data_path, "registration")

    # Grab all the nii.gz files
    nii_gz_files = glob_files(registration_folder, "nii.gz")

    # Sort the files according to the number at the end
    nii_gz_files = sorted(nii_gz_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Print the number of files
    print("Found {} nii files".format(len(nii_gz_files)))

    # Perform the concatenation
    concat_files(data_path, nii_gz_files)

if __name__ == "__main__":
    main()
