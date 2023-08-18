import sys
from py_helpers.general_helpers import *
import numpy as np

import nibabel as nib

from nibabel.affines import apply_affine

import numpy.linalg as npl


# Function to do the streamline node extraction
def dwi_extraction(nii_bvec_bval_mif, output_path, K=8):
    
    # Create region folder
    region_folder = os.path.join(output_path, nii_bvec_bval_mif[0])

    # Check if region folder exists
    check_output_folders(region_folder, "region_folder", wipe=False)

    # Load the nii file
    nii = nib.load(nii_bvec_bval_mif[1])

    # Slice it to the first 8 shells
    nii_data = nii.slicer[:, :, :, :K]

    # Define the new file name
    new_dwi_name = nii_bvec_bval_mif[1].split(os.sep)[-1].replace(".nii", "_extracted.nii")

    # Define new filepath
    new_dwi_path = os.path.join(region_folder, new_dwi_name)

    # Save the new file
    nib.save(nii_data, new_dwi_path)

    # Extract the bvecs and bvals
    bvec_text = np.loadtxt(nii_bvec_bval_mif[2])
    bval_text = np.loadtxt(nii_bvec_bval_mif[3])

    # Slice them to the first 8 shells
    bvec_shell = bvec_text[:, :K]
    bval_shell = bval_text[:K]

    # Define the new file name
    new_bvec_name = nii_bvec_bval_mif[2].split(os.sep)[-1].replace(".bvecs", "_extracted.bvecs")
    new_bval_name = nii_bvec_bval_mif[3].split(os.sep)[-1].replace(".bvals", "_extracted.bvals")

    # Define new filepath
    new_bvec_path = os.path.join(region_folder, new_bvec_name)
    new_bval_path = os.path.join(region_folder, new_bval_name)

    # Save the new files
    np.savetxt(new_bvec_path, bvec_shell)
    np.savetxt(new_bval_path, bval_shell)

    # Convert to mif
    # MIF_CMD = 


# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    if hpc:
        data_path = "/rds/general/user/hsa22/ephemeral/Brain_MINDS/processed_dMRI/dMRI_unzipped"
        output_path = "/rds/general/user/hsa22/ephemeral/Brain_MINDS/processed_dMRI/extracted_dwi_originalsize"
    else:
        data_path = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/trk_data_originalsize"
        template = "/media/hsa22/Expansion/Brain-MINDS/unused_model_data/dMRI_b0/A10-R01_0028-TT21/DWI_concatenated_b0.nii.gz"
        output_path = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/trk_data_originalsize_voxels"

    check_output_folders(output_path, "output_path", wipe=False)

    # Grab all the nii, bvecs and bval files + mif files
    nii_files = glob_files(data_path, "nii")
    bvec_files = glob_files(data_path, "bvecs")
    bval_files = glob_files(data_path, "bvals")
    mif_files = glob_files(data_path, "mif")

    # Make sure we only keep ones with concatenated
    nii_files = [nii for nii in nii_files if "concatenated" in nii
                 and "b0" not in nii and "extracted" not in nii
                 and "resized" not in nii]
    bvec_files = [bvec for bvec in bvec_files if "concatenated" in bvec
                    and "b0" not in bvec and "extracted" not in bvec
                    and "resized" not in bvec]
    bval_files = [bval for bval in bval_files if "concatenated" in bval
                    and "b0" not in bval and "extracted" not in bval
                    and "resized" not in bval]
    mif_files = [mif for mif in mif_files if "concatenated" in mif
                    and "b0" not in mif and "extracted" not in mif
                    and "resized" not in mif]

    # Create list of [region_ID, nii, bvec, bval]
    nii_bvec_bval_mif = []
    for nii in nii_files:
        region_ID = nii.split(os.sep)[-3]
        bvec = [bvec for bvec in bvec_files if region_ID in bvec][0]
        bval = [bval for bval in bval_files if region_ID in bval][0]
        mif = [mif for mif in mif_files if region_ID in mif][0]
        nii_bvec_bval_mif.append([region_ID, nii, bvec, bval, mif])

    print("Found {} nii files".format(len(nii_bvec_bval_mif)))

    # Get which region to run
    if hpc:
        file_idx = int(sys.argv[2])
        dwi_extraction(nii_bvec_bval_mif[file_idx], output_path)
    else:
        for file_idx in range(len(nii_bvec_bval_mif)):
            dwi_extraction(nii_bvec_bval_mif[file_idx], output_path)

if __name__ == "__main__":
    main()