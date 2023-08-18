import sys
from py_helpers.general_helpers import *
import numpy as np

import subprocess

# Function to do the streamline node extraction
def dwi_extraction(nii_bvec_bval):
    
    # Create the mif file name
    mif_name = nii_bvec_bval[1].split(os.sep)[-1].replace(".nii", ".mif")

    # Create the mif file path
    mif_path = os.path.join((os.sep).join(nii_bvec_bval[1].split(os.sep)[:-1]), mif_name)

    # Convert to mif command
    MIF_CMD = "mrconvert {input_nii} -fslgrad {bvec} {bval} {output}".format(input_nii=nii_bvec_bval[1], 
                                                                             bvec=nii_bvec_bval[2], 
                                                                             bval=nii_bvec_bval[3], 
                                                                             output=mif_path)
    
    # Run the command
    print("Running: {}".format(MIF_CMD))
    subprocess.run(MIF_CMD, shell=True, check=True)

# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    if hpc:
        data_path = "/rds/general/user/hsa22/ephemeral/Brain_MINDS/processed_dMRI/extracted_dwi_originalsize"
    else:
        data_path = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/trk_data_originalsize"
        template = "/media/hsa22/Expansion/Brain-MINDS/unused_model_data/dMRI_b0/A10-R01_0028-TT21/DWI_concatenated_b0.nii.gz"
        output_path = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/trk_data_originalsize_voxels"

    # Grab all the nii, bvecs and bval files + mif files
    nii_files = glob_files(data_path, "nii")
    bvec_files = glob_files(data_path, "bvecs")
    bval_files = glob_files(data_path, "bvals")

    # Create list of [region_ID, nii, bvec, bval]
    nii_bvec_bval = []
    for nii in nii_files:
        region_ID = nii.split(os.sep)[-3]
        bvec = [bvec for bvec in bvec_files if region_ID in bvec][0]
        bval = [bval for bval in bval_files if region_ID in bval][0]
        nii_bvec_bval.append([region_ID, nii, bvec, bval])

    print("Found {} nii files".format(len(nii_bvec_bval)))

    # Get which region to run
    if hpc:
        file_idx = int(sys.argv[2])
        dwi_extraction(nii_bvec_bval[file_idx])
    else:
        for file_idx in range(len(nii_bvec_bval)):
            dwi_extraction(nii_bvec_bval[file_idx])

if __name__ == "__main__":
    main()