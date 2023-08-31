import sys
from py_helpers.general_helpers import *
import nibabel as nib

import subprocess

# Function to register the fmri data
def register_fmri(fmri_file, output_path, atlas_path, transform_path):

    # Get the file name
    fmri_name = fmri_file.split(os.sep)[-1].split(".")[0]

    # Create the output directory
    fmri_folder = os.path.join(output_path, fmri_name)
    check_output_folders(fmri_folder, "fmri output", wipe=False)

    # Slices folder
    slices_folder = os.path.join(fmri_folder, "slices")
    check_output_folders(slices_folder, "slices output", wipe=False)

    # Registration folder
    registration_folder = os.path.join(fmri_folder, "registration")
    check_output_folders(registration_folder, "registration output", wipe=False)

    # Load the fmri file
    fmri_img = nib.load(fmri_file)

    # Get the number of volumes
    num_vols = fmri_img.shape[-1]

    # Loop through the volumes
    for vol_idx in range(num_vols):

        # Get the volume
        fmri_slice = fmri_img.slicer[:, :, :, vol_idx]

        # Get the file path for slices
        slices_filepath = os.path.join(slices_folder, "slice_{idx}".format(idx=vol_idx))
        slices_filepath_with_ext = slices_filepath + ".nii.gz"

        # Get the file path for registration
        registration_filepath = os.path.join(registration_folder, "slice_{idx}".format(idx=vol_idx))

        # Save the slice if it doesn't exist
        if not os.path.exists(slices_filepath_with_ext):
            nib.save(fmri_slice, slices_filepath_with_ext)
            print("Saved slice {}".format(vol_idx))

        # Run the registration if the file doesn't exist
        if not os.path.exists(registration_filepath):
            # Register the slice to the atlas using the ants command
            ANTS_CMD = "antsApplyTransforms -i {fmri} -r {ref} -o {filepath}.nii.gz -t {transform} -n NearestNeighbor".format(
                fmri=slices_filepath_with_ext, ref=atlas_path, filepath=registration_filepath, transform=transform_path
            )

            # Run the command
            print("Running ANTS command for slice {}".format(vol_idx))
            subprocess.run(ANTS_CMD, shell=True)


    # Grab all the slices in the registration
    registered_slices = glob_files(registration_folder, "nii.gz")

    # Define all slices in one string
    registered_slices_str = " ".join(registered_slices)

    # Define the output file path
    output_filepath = os.path.join(fmri_folder, "registered_fmri.nii.gz")

    # Merge the slices into one file if it doesn't exist
    if not os.path.exists(output_filepath):
        CONCAT_CMD = "mrcat {slices} {output}".format(slices=registered_slices_str, output=output_filepath)
        # Run the command
        print("Running concat command")
        subprocess.run(CONCAT_CMD, shell=True)

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
        reference_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/BMCR_STPT_template/Atlases/MBM_mapped/MBM_cortex_vM_80um_TC_std.nii.gz"
        transform_path = "/rds/general/ephemeral/user/hsa22/ephemeral/MBM_fmri/MBM_2_STPT.h5"
    else:
        data_path = "/mnt/d/MBM_fmri/sub-NIHm32/ses-01"
        reference_path = "/mnt/d/Brain-MINDS/BMCR_STPT_template/Atlases/MBM_mapped/MBM_cortex_vM_80um_TC_std.nii.gz"
        transform_path = "/mnt/d/MBM_fmri/MBM_2_STPT.h5"

    # Grab all the nii.gz files
    nii_gz_files = glob_files(data_path, "nii.gz")

    # Filter for the non-slices 
    nii_gz_files = [file for file in nii_gz_files if "slices" not in file]

    print("Found {} nii files".format(len(nii_gz_files)))

    # Create the output directory
    output_path = os.path.join(data_path, "fmri_slices")
    check_output_folders(output_path, "main output", wipe=False)

    # Run the commands
    if hpc:
        file_idx = int(sys.argv[2])
        register_fmri(nii_gz_files[file_idx], output_path, reference_path, transform_path)
    else:
        for file_idx in range(len(nii_gz_files)):
            register_fmri(nii_gz_files[file_idx], output_path, reference_path, transform_path)


if __name__ == "__main__":
    main()
