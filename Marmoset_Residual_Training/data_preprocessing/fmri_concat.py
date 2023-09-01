import sys
from py_helpers.general_helpers import *
import nibabel as nib

import numpy as np

import subprocess

# Function to concatenate the files
def concat_files(data_path, nii_gz_files):

    """Concatenate NIfTI files using mrcat from MRtrix3."""

    # Create new file path
    registered_fmri_path = os.path.join(data_path, "concat_registered_fmri.nii.gz")
    
    # Start by initializing the output file with the first image
    subprocess.run(['cp', nii_gz_files[0], registered_fmri_path])

    # Now, for each subsequent file, concatenate it with the output file
    for i in range(1, len(nii_gz_files)):
        print("-"*50)
        print("Concatenating file {} of {}".format(i, len(nii_gz_files)))
        temp_output = f'temp_concat_{i}.nii.gz'
        subprocess.run(['mrcat', registered_fmri_path, nii_gz_files[i], temp_output, '-force'])
        
        # Move the temporary output file to the main output file
        subprocess.run(['mv', temp_output, registered_fmri_path])
    

# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    if hpc:
        data_path = "/rds/general/user/hsa22/ephemeral/MBM_fmri/sub-NIHm32/ses-01/fmri_slices/sub-NIHm32_ses-01_task-rest_run-LR-2"
    else:
        data_path = "/media/hsa22/Expansion/MBM_fmri/sub-NIHm32/ses-01/fmri_slices/sub-NIHm32_ses-01_task-rest_run-LR-2"

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
