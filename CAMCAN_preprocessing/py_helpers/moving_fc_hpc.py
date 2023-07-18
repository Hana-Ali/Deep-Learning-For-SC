import os
import argparse
import sys

sys.path.append("..")
from py_helpers.general_helpers import *

from collections import defaultdict

import shutil

# Define the path to the mrtrix data in the HPC
path_to_outputs = "/rds/general/user/hsa22/ephemeral/CAMCAN/dMRI_outputs"

# Define the path to the fmri data
fmri_path = "/rds/general/user/hsa22/ephemeral/CAMCAN/camcan_parcellated_acompcor/glasser360/fmri700/rest"

# Function to move data
def move_data(path_to_connectivities, path_to_fmri):

    # Grab all the csvs in connectivities
    csv_files = glob_files(path_to_connectivities, "csv")

    # Create a dictionary, where the keys are the subject IDs and the values are the paths to the CSV files
    csv_dict = defaultdict(list)
    # Loop through the CSV files
    for csv_file in csv_files:
        # Get the subject ID
        subject_ID = csv_file.split(os.sep)[-3]
        # Append the path to the CSV file to the dictionary, using the subject ID as the key
        csv_dict[subject_ID].append(csv_file)

    # Grab the .mat fmri files
    mat_files = glob_files(fmri_path, "mat")

    # Keep only the mat files that are in the csv_dict
    mat_files = [mat_file for mat_file in mat_files if mat_file.split(os.sep)[-2] in csv_dict.keys()]

    # Ensure that the number of mat files is the same as the number of keys in the csv_dict
    assert len(mat_files) == len(csv_dict.keys())

    # Make path to fmri, if it's not passed as an argument
    if not path_to_fmri:
        path_to_fmri = os.path.join(path_to_outputs, "fmri")

    # Check that the folder exists
    check_output_folders(path_to_fmri, "fmri", wipe=False)

    # For each mat file
    for mat_file in mat_files:
        # Get the filename
        filename = mat_file.split(os.sep)[-1]
        # Get the subject name
        subject_ID = filename.split("_")[0]
        # Create a folder for the subject
        path_to_subject = os.path.join(path_to_fmri, subject_ID)
        check_output_folders(path_to_subject, subject_ID, wipe=False)
        # Create the actual file path
        new_filename = os.path.join(path_to_subject, filename)
        # Copy the file
        shutil.copy(mat_file, new_filename)


if __name__ == "__main__":
    # Get the parser
    parser = argparse.ArgumentParser(description="Move the mrtrix data to the HPC")

    # Add the arguments
    parser.add_argument("--path_to_connectivities", type=str, default=os.path.join(path_to_outputs, "connectivities"), 
                                                help="Path to save connectivities to")
    parser.add_argument("--path_to_fmri", type=str, default=os.path.join(path_to_outputs, "fmri"),
                                                help="Path to save fmri data to")
    
    # Parse the arguments
    args = parser.parse_args()

    # Move the data
    move_data(args.path_to_connectivities, args.path_to_fmri)



