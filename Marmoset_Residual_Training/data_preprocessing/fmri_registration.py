import sys
from py_helpers.general_helpers import *
import nibabel as nib

import subprocess

# Function to register the fmri data
def register_fmri(fmri_file, output_path, atlas_path):

    # Get the file name
    fmri_name = fmri_file.split(os.sep)[-1].split(".")[0]

    # Create the output directory
    fmri_folder = os.path.join(output_path, fmri_name)
    check_output_folders(fmri_folder, "fmri output", wipe=False)

    # Load the fmri file
    fmri_img = nib.load(fmri_file)

    # Get the number of volumes
    num_vols = fmri_img.shape[-1]

    # Loop through the volumes
    for vol_idx in range(num_vols):

        # Get the volume
        fmri_slice = fmri_img.slicer[:, :, :, vol_idx]

        # Get the file path
        filepath = os.path.join(fmri_folder, "{name}_slice_{idx}".format(name=fmri_name, idx=vol_idx))

        # Save the slice
        nib.save(fmri_slice, filepath)

        print("Saved slice {}".format(vol_idx))

        # # Register the slice to the atlas using the flirt command
        # FLIRT_CMD = "flirt -in {fmri} -ref {atlas} -out {filepath} -omat {filepath}.mat".format(
        #     fmri=fmri_slice, atlas=atlas_path, filepath=filepath
        # )

        # # Run the command
        # print("Running command")
        # subprocess.run(FLIRT_CMD, shell=True)
    

# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    if hpc:
        data_path = "/rds/general/ephemeral/user/hsa22/ephemeral/MBM_fmri/sub-NIHm32/ses-01"
        atlas_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/BMCR_STPT_template/Atlases/MBM_mapped/MBM_cortex_vPaxinos_80um_TC_std.nii.gz"
    else:
        data_path = "/mnt/d/MBM_fmri/sub-NIHm32/ses-01"
        atlas_path = "/mnt/d/Brain-MINDS/BMCR_STPT_template/Atlases/MBM_mapped/MBM_cortex_vPaxinos_80um_TC_std.nii.gz"

    # Grab all the nii.gz files
    nii_gz_files = glob_files(data_path, "nii.gz")

    print("Found {} nii files".format(len(nii_gz_files)))

    # Create the output directory
    output_path = os.path.join(data_path, "fmri_slices")
    check_output_folders(output_path, "main output", wipe=False)

    # Run the commands
    if hpc:
        file_idx = int(sys.argv[2])
        register_fmri(nii_gz_files[file_idx], output_path, atlas_path)
    else:
        for file_idx in range(len(nii_gz_files)):
            register_fmri(nii_gz_files[file_idx], output_path, atlas_path)


if __name__ == "__main__":
    main()
