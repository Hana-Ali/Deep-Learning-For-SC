import sys
from py_helpers.general_helpers import *
import numpy as np
import shutil

# Function to do the streamline node extraction
def remove_first_item(npy_file):

    # Load the npy file
    npy_data = np.load(npy_file)

    # Make a new numpy array to store the sliced data
    npy_data_sliced = np.zeros((npy_data.shape[0], npy_data.shape[1] - 1, npy_data.shape[2]))

    # From every streamline, remove the first item
    for streamline_idx in range(len(npy_data)):
        npy_data_sliced[streamline_idx] = npy_data[streamline_idx][1:]

    # Get the filename of the npy file
    new_filename = os.path.join((os.sep).join(npy_file.split(os.sep)[:-1]), npy_file.split(os.sep)[-1].replace("points_direction", "points_direction_no_first"))

    # Save the new npy file
    np.save(new_filename, npy_data_sliced)

    print("Saved file: ", new_filename, " with shape: ", npy_data_sliced.shape)

# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    if hpc:
        data_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/model_data_w_resize/only_points_direction"
    else:
        data_path = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/only_points_direction"

    # Grab the  numpy files
    npy_files = glob_files(data_path, "npy")

    # Filter for ones with "points_direction" in name
    points_direction = [file for file in npy_files if "points_direction" in file]

    print("Number of files: ", len(points_direction))

    # Get which region to run
    if hpc:
        file_idx = int(sys.argv[2])
        remove_first_item(points_direction[file_idx])
    else:
        for file_idx in range(len(points_direction)):
            remove_first_item(points_direction[file_idx])

if __name__ == "__main__":
    main()