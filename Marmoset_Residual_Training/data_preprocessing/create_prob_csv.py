import sys
from py_helpers.general_helpers import *
import numpy as np

import subprocess

# Function to do the connectome creation
def connectome_creation(tck_file, connectome_folder, atlas_reg_paths):

    # Get the region ID
    region_ID = tck_file.split(os.sep)[-3]

    # Create the region folder
    region_folder = os.path.join(connectome_folder, region_ID)
    check_output_folders(region_folder, "region", wipe=False)

    # Create the connectome file name
    connectome_name = tck_file.split(os.sep)[-1].replace(".tck", "_tracer.csv")

    # Create the connectome file path
    connectome_path = os.path.join(region_folder, connectome_name)

    # Connectivity matrix command
    CONNECTIVITY_PROB_CMD = "tck2connectome {input} {atlas} {output} -zero_diagonal -symmetric \
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
        tck_path = "/media/hsa22/Expansion/Brain-MINDS/BMCR_core_data/meta_data"
        mrtrix_path = "/media/hsa22/Expansion/Brain-MINDS/processed_dMRI/MRTRIX"
        atlas_reg_path = "/media/hsa22/Expansion/Brain-MINDS/BMCR_STPT_template/Atlases/registered_atlas_integer.nii.gz"

    # Grab all the tck files
    tck_files = glob_files(tck_path, "tck")

    # Filter out for the tracer streamline ones
    tck_files = [tck for tck in tck_files if "tracer" in tck and "sharp" not in tck]

    # Create connectome folder in mrtrix folder
    connectome_folder = os.path.join(mrtrix_path, "connectomes_MBCA_tracer")
    check_output_folders(connectome_folder, "connectome", wipe=False)

    # Get which region to run
    if hpc:
        file_idx = int(sys.argv[2])
        connectome_creation(tck_files[file_idx], connectome_folder, atlas_reg_path)
    else:
        for file_idx in range(len(tck_files)):
            connectome_creation(tck_files[file_idx], connectome_folder, atlas_reg_path)


if __name__ == "__main__":
    main()
    # tck_path = "/media/hsa22/Expansion/Brain-MINDS/combined_streamlines/combined_tracer_streamlines.tck"
    # atlas_reg_path = "/media/hsa22/Expansion/Brain-MINDS/BMCR_STPT_template/Atlases/registered_atlas_integer.nii.gz"
    # connectome_path = "/media/hsa22/Expansion/Brain-MINDS/combined_streamlines/combined_tracer_streamlines.csv"

    # CONNECTIVITY_PROB_CMD = "tck2connectome {input} {atlas} {output} -zero_diagonal -symmetric \
    # -assignment_all_voxels -force".format(input=tck_path, atlas=atlas_reg_path, output=connectome_path)

    # # Run the command
    # print("Running: {}".format(CONNECTIVITY_PROB_CMD))
    # subprocess.run(CONNECTIVITY_PROB_CMD, shell=True, check=True)
