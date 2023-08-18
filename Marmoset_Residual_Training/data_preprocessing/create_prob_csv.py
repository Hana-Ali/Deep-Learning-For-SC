import sys
from py_helpers.general_helpers import *
import numpy as np

import subprocess

# Function to do the connectome creation
def connectome_creation(tck_file, atlas_reg_paths):

    # Get the file path to create the output folder
    output_path = (os.sep).join(tck_file.split(os.sep)[:-2])

    # Create the connectome folder
    connectome_folder = os.path.join(output_path, "connectomes_MBCA")
    check_output_folders(connectome_folder, "connectome", wipe=False)

    # Create the connectome file name
    connectome_name = tck_file.split(os.sep)[-1].replace(".tck", ".csv")

    # Create the connectome file path
    connectome_path = os.path.join(connectome_folder, connectome_name)

    # Connectivity matrix command
    CONNECTIVITY_PROB_CMD = "tck2connectome {input} {atlas}.nii.gz {output} -zero_diagonal -symmetric \
        -assignment_all_voxels -force".format(input=tck_file, atlas=atlas_reg_paths, output=connectome_path)
    
    # Run the command
    print("Running: {}".format(CONNECTIVITY_PROB_CMD))
    subprocess.run(CONNECTIVITY_PROB_CMD, shell=True, check=True)


# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    if hpc:
        tck_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/processed_dMRI/MRTRIX"
    else:
        tck_path = "/media/hsa22/Expansion/Brain-MINDS/processed_dMRI/MRTRIX"
        atlas_reg_path = "/media/hsa22/Expansion/Brain-MINDS/BMCR_STPT_template/Atlases/registered_MBCA_atlas"

    # Grab all the tck files
    tck_files = glob_files(tck_path, "tck")

    # Get which region to run
    if hpc:
        file_idx = int(sys.argv[2])
        connectome_creation(tck_files[file_idx], atlas_reg_path)
    else:
        for file_idx in range(len(tck_files)):
            connectome_creation(tck_files[file_idx], atlas_reg_path)


if __name__ == "__main__":
    main()
