import sys
from py_helpers.general_helpers import *
import numpy as np

import subprocess

# Function to do the streamline node extraction
def peaks_extraction(fod_file, new_output_folder):
    
    # Get region folder name
    region_ID = fod_file.split(os.sep)[-2]
    region_folder = os.path.join(new_output_folder, region_ID)
    check_output_folders(region_folder, "region_folder", wipe=False)

    # Get the new filename for the peak from the fod file
    peaks_file = ("_").join(fod_file.split(os.sep)[-1].split("_")[:2]) + "_peaks.nii.gz"

    # Define the output file
    peaks_path = os.path.join(region_folder, peaks_file)

    # Define the command
    cmd = "sh2peaks {fod} {output}".format(fod=fod_file, output=peaks_path)

    # Run the command
    print("Running: {}".format(cmd))
    subprocess.run(cmd, shell=True, check=True)

# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    if hpc:
        data_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/model_data/MRTRIX_fod_norms"
        new_output_folder = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/model_data/MRTRIX_fod_peaks"
    else:
        data_path = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/MRTRIX_fod_norms"
        new_output_folder = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/MRTRIX_fod_peaks"

    # Grab the nii.gz files
    nii_gz_files = glob_files(data_path, "nii.gz")

    # Get which region to run
    file_idx = int(sys.argv[2])
    peaks_extraction(nii_gz_files[file_idx], new_output_folder)

if __name__ == "__main__":
    main()