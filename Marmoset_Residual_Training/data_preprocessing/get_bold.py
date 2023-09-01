import sys
import os
from py_helpers.general_helpers import *
from nilearn.input_data import NiftiLabelsMasker
import numpy as np

# Function to concatenate the files
def get_bold(nii_file, data_path, MBM_path, MBCA_path):

    # Define the output directory
    output_folder = os.path.join(data_path, "BOLD_slices")
    check_output_folders(output_folder, "BOLD output", wipe=False)

    # Define the output folders for each atlas
    MBM_output = os.path.join(output_folder, "MBM")
    MBCA_output = os.path.join(output_folder, "MBCA")
    check_output_folders(MBM_output, "MBM output", wipe=False)
    check_output_folders(MBCA_output, "MBCA output", wipe=False)

    # For both atlases
    for atlas_path in [MBM_path, MBCA_path]:

        # Get the masker object
        masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=True)

        # Get the timeseries BOLD measure
        time_series = masker.fit_transform(nii_file)

        # Get the file name
        file_name = nii_file.split(os.sep)[-1].split(".")[0] + ".npy"

        # If MBM
        if "MBM" in atlas_path:
            output_file = os.path.join(MBM_output, file_name)
        else:
            output_file = os.path.join(MBCA_output, file_name)

        # Save the file
        np.save(output_file, time_series)


# Main function
def main():

    # Define the path to the data
    hpc = False
    labs = True
    if hpc:
        data_path = "/rds/general/user/hsa22/ephemeral/MBM_fmri/sub-NIHm32/ses-01/fmri_slices/sub-NIHm32_ses-01_task-rest_run-LR-2"
        MBM_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/BMCR_STPT_template/Atlases/MBM_mapped/MBM_cortex_vM_80um_TC_std.nii.gz"
        MBCA_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/BMCR_STPT_template/Atlases/MBCA_mapped/atlas_segmentation.nii.gz"
    elif labs:
        data_path = "/media/hsa22/Expansion/MBM_fmri/sub-NIHm32/ses-01/fmri_slices/sub-NIHm32_ses-01_task-rest_run-LR-2"
        MBM_path = "/media/hsa22/Expansion/Brain-MINDS/BMCR_STPT_template/Atlases/MBM_mapped/MBM_cortex_vM_80um_TC_std.nii.gz"
        MBCA_path = "/media/hsa22/Expansion/Brain_MINDS/BMCR_STPT_template/Atlases/MBCA_mapped/atlas_segmentation.nii.gz"
    else:
        data_path = "/mnt/d/MBM_fmri/sub-NIHm32/ses-01/fmri_slices/sub-NIHm32_ses-01_task-rest_run-LR-2"
        MBM_path = "/mnt/d/Brain-MINDS/BMCR_STPT_template/Atlases/MBM_mapped/MBM_cortex_vM_80um_TC_std.nii.gz"
        MBCA_path = "/mnt/d/Brain_MINDS/BMCR_STPT_template/Atlases/MBCA_mapped/atlas_segmentation.nii.gz"

    # Define the folder with the registered slices
    registration_folder = os.path.join(data_path, "registration")

    # Grab all the nii.gz files from registration
    nii_gz_files = glob_files(registration_folder, "nii.gz")

    # Print the number of files
    print("Found {} nii files".format(len(nii_gz_files)))

    # Run the commands
    if hpc:
        file_idx = int(sys.argv[2])
        get_bold(nii_gz_files[file_idx], data_path, MBM_path, MBCA_path)
    else:
        for file_idx in range(len(nii_gz_files)):
            get_bold(nii_gz_files[file_idx], data_path, MBM_path, MBCA_path)

    # Get the BOLD for each

if __name__ == "__main__":
    main()
