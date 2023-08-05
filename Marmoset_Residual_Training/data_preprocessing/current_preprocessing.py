import sys
from py_helpers.general_helpers import *
import numpy as np

import subprocess

# Function to do the streamline node extraction
def streamline_node_extraction(tck_file, template, new_output_folder):
    
    # Get region folder name
    region_ID = tck_file.split(os.sep)[-2]
    region_folder = os.path.join(new_output_folder, region_ID)
    check_output_folders(region_folder, "region_folder", wipe=False)

    # Get the new folder name for this streamline
    streamline_folder = tck_file.split(os.sep)[-1].replace(".tck", "_node_voxel_locations")

    # Define the output folder
    output_folder = os.path.join(region_folder, streamline_folder)
    check_output_folders(output_folder, "output_folder for tck voxels", wipe=False)

    # Define the command
    cmd = "tckconvert -scanner2voxel {template} {input} {output_folder}/streamline-'[]'.txt".format(input=tck_file, template=template, output_folder=output_folder)

    # Run the command
    print("Running: {}".format(cmd))
    subprocess.run(cmd, shell=True, check=True)

# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    if hpc:
        data_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/model_data_w_resize/resampled_streamlines"
        template = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/model_data_w_resize/dMRI_b0/A10-R01_0028-TT21/DWI_concatenated_b0_resized.nii.gz"
        new_output_folder = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/model_data_w_resize/streamline_nodes"
    else:
        data_path = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/resampled_streamlines"
        template = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/dMRI_b0/A10-R01_0028-TT21/DWI_concatenated_b0_resized.nii.gz"
        new_output_folder = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/streamline_nodes"

    # Grab the tck files - should be 156 (3 types x 52 injections)
    tck = glob_files(data_path, "tck")

    # Get which region to run
    file_idx = int(sys.argv[2])
    streamline_node_extraction(tck[file_idx], template, new_output_folder)

if __name__ == "__main__":
    main()