import sys
sys.path.append("..")
from py_helpers.general_helpers import *
import os

import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Define the path to the mrtrix data in the HPC
path_to_outputs = "/rds/general/user/hsa22/ephemeral/CAMCAN/dMRI_outputs"

# Function to move data
def move_data(path_to_connectivities):
    # Define the path to the mrtrix data
    path_to_mrtrix = os.path.join(path_to_outputs, "mrtrix")

    # Grab all CSV files in the mrtrix folder
    csv_files = glob_files(path_to_mrtrix, "csv")

    # Create a dictionary, where the keys are the subject IDs and the values are the paths to the CSV files
    csv_dict = defaultdict(list)
    # Loop through the CSV files
    for csv_file in csv_files:
        # Get the subject ID
        subject_ID = csv_file.split(os.sep)[-4]
        # Append the path to the CSV file to the dictionary, using the subject ID as the key
        csv_dict[subject_ID].append(csv_file)

    print("Number of subjects: ", len(csv_dict.keys()))

    # Make path for connectivities, if it's not passed as an argument
    if not path_to_connectivities:
        path_to_connectivities = os.path.join(path_to_outputs, "connectivities")
    
    # Check that the folder exists
    check_output_folders(path_to_connectivities, "connectivities", wipe=False)

    # For each subject
    for subject_ID, file in csv_dict.items():
        print("Subject ID: ", subject_ID)
        # Make a folder for the subject
        path_to_subject = os.path.join(path_to_connectivities, subject_ID)
        check_output_folders(path_to_subject, subject_ID, wipe=False)
        # Make a folder for probabilistic and global
        path_to_probabilistic = os.path.join(path_to_subject, "probabilistic")
        check_output_folders(path_to_probabilistic, "probabilistic", wipe=False)
        path_to_global = os.path.join(path_to_subject, "global")
        check_output_folders(path_to_global, "global", wipe=False)
        # Save the csv files to the subject folder
        for csv_file in file:
            # Get the name of the csv file
            csv_name = csv_file.split(os.sep)[-1]
            # Get whether it's probabilistic or global
            if "prob" in csv_file:
                new_filename = os.path.join(path_to_probabilistic, csv_name)
            elif "global" in csv_file:
                new_filename = os.path.join(path_to_global, csv_name)
            else:
                return ValueError("Couldn't find probabilistic or global in the filename")
            # Copy the csv file to the subject folder
            shutil.copy(csv_file, new_filename)

if __name__ == "__main__":
    # Get the parser
    parser = argparse.ArgumentParser(description="Move the mrtrix data to the HPC")

    # Add the arguments
    parser.add_argument("--path_to_connectivities", type=str, default=os.path.join(path_to_outputs, "connectivities"), 
                                                help="Path to save connectivities to")
    parser.add_argument("--move", action="store_true", help="Move the data")

    # Parse the arguments
    args = parser.parse_args()

    # If we want to move the data
    if args.move:
        move_data(args.path_to_connectivities)

    # Grab a random csv file from the connectivities folder
    csv_file = glob_files(args.path_to_connectivities, "csv")[0]

    # Load the csv file with numpy
    csv_data = np.loadtxt(csv_file, delimiter=",")
    print("Shape of csv data: ", csv_data.shape)
    print("Name of csv file: ", csv_file)

    # Plot the data
    plt.imshow(csv_data, cmap='jet', aspect='equal', interpolation='nearest')
    cb = plt.colorbar()
    cb.set_label('Connection strength')

    # Save the plot
    plt.savefig("test.png")
