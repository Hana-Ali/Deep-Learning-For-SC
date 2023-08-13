import sys
from py_helpers.general_helpers import *

import shutil

# Function to do the streamline node extraction
def find_angle_direction_files(npy_file, output_path):

    # Get the region folder name
    region_ID = npy_file.split(os.sep)[-3]

    # Get whether it's a trk or tck file
    filetype = npy_file.split(os.sep)[-2]

    # Define the trk and tck folders for this region
    region_folder = os.path.join(output_path, region_ID)
    check_output_folders(region_folder, "region_folder", wipe=False)
    
    # Define the trk and tck folders for this region
    filetype_folder = os.path.join(region_folder, filetype)
    check_output_folders(filetype_folder, "filetype_folder", wipe=False)

    # Get the filename of the npy file
    direction_filename = npy_file.split(os.sep)[-1]

    # Define the new filepath
    direction_filepath = os.path.join(filetype_folder, direction_filename)

    # Copy the npy file to the new location
    shutil.copyfile(npy_file, direction_filepath)

    print("Copied file: ", direction_filepath)


# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    if hpc:
        data_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/model_data_w_resize/streamline_stats"
        output_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/model_data_w_resize/only_points_direction"
    else:
        data_path = "/media/hsa22/Expansion/Brain-MINDS/streamline_stats"
        output_path = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/only_points_direction"

    check_output_folders(output_path, "output_path", wipe=False)

    # Grab the  numpy files
    npy_files = glob_files(data_path, "npy")

    # Filter for ones with "points_direction" in name
    points_direction = [file for file in npy_files if "points_direction" in file]

    print("Number of files: ", len(points_direction))

    # Get which region to run
    if hpc:
        file_idx = int(sys.argv[2])
        find_angle_direction_files(points_direction[file_idx], output_path)
    else:
        for file_idx in range(len(points_direction)):
            find_angle_direction_files(points_direction[file_idx], output_path)

if __name__ == "__main__":
    main()