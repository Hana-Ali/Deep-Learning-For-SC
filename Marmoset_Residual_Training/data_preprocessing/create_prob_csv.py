import sys
from py_helpers.general_helpers import *
import numpy as np

import subprocess

# Function to do the connectome creation
def connectome_creation(tck_file, atlas_dictionary):

    # Get the region ID
    region_ID = tck_file.split(os.sep)[-3]

    # For every atlas
    for atlas in atlas_dictionary.keys():

        # Get the atlas path
        atlas_path = atlas_dictionary[atlas][0]

        # Get the connectome folder
        connectome_folder = atlas_dictionary[atlas][1]

        # Create the region folder
        region_folder = os.path.join(connectome_folder, region_ID)
        check_output_folders(region_folder, "region", wipe=False)
            
        # Create the connectome file name
        connectome_name = tck_file.split(os.sep)[-1].replace(".tck", "_tracer.csv")

        # Create the connectome file path
        connectome_path = os.path.join(region_folder, connectome_name)

        # Connectivity matrix command
        CONNECTIVITY_COMMAND = "tck2connectome {input} {atlas} {output} -zero_diagonal -symmetric \
            -assignment_all_voxels -force".format(input=tck_file, atlas=atlas_path, output=connectome_path)
        
        # Run the command
        print("Running: {}".format(CONNECTIVITY_COMMAND))
        subprocess.run(CONNECTIVITY_COMMAND, shell=True, check=True)


# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    if hpc:
        tck_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/processed_dMRI/MRTRIX"
    else:
        tck_path = "/media/hsa22/Expansion/Brain-MINDS/BMCR_core_data/meta_data"
        mrtrix_path = "/media/hsa22/Expansion/Brain-MINDS/processed_dMRI/MRTRIX"
        atlas_reg_path = "/media/hsa22/Expansion/Brain-MINDS/BMCR_STPT_template/Atlases"

    # Grab all the tck files
    tck_files = glob_files(tck_path, "tck")

    # Filter out for the tracer streamline ones
    tck_files = [tck for tck in tck_files if "tracer" in tck and "sharp" not in tck]

    # Grab all the nii.gz files (atlas)
    atlas_reg_paths = glob_files(atlas_reg_path, "nii.gz")

    # Get the BMA, MBCA, and MBM atlas paths
    BMA_atlas_path = [atlas for atlas in atlas_reg_paths if "BMA_mapped" in atlas][0]
    MBCA_atlas_path = [atlas for atlas in atlas_reg_paths if "MBCA_mapped" in atlas 
                       and "segmentation" in atlas][0]
    MBM_atlas_path = [atlas for atlas in atlas_reg_paths if "MBM_mapped" in atlas
                      and "Paxinos" in atlas][0]
    
    # Make connectome folders
    BMA_folder = os.path.join(mrtrix_path, "BMA_connectomes")
    MBCA_folder = os.path.join(mrtrix_path, "MBCA_connectomes")
    MBM_folder = os.path.join(mrtrix_path, "MBM_connectomes")
    check_output_folders(BMA_folder, "BMA", wipe=False)
    check_output_folders(MBCA_folder, "MBCA", wipe=False)
    check_output_folders(MBM_folder, "MBM", wipe=False)

    # Make a dictionary of the atlas paths and folders
    atlas_dictionary = {
        "BMA": [BMA_atlas_path, BMA_folder],
        "MBCA": [MBCA_atlas_path, MBCA_folder],
        "MBM": [MBM_atlas_path, MBM_folder]
    }


    # Get which region to run
    if hpc:
        file_idx = int(sys.argv[2])
        connectome_creation(tck_files[file_idx], atlas_dictionary)
    else:
        for file_idx in range(len(tck_files)):
            connectome_creation(tck_files[file_idx], atlas_dictionary)


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
