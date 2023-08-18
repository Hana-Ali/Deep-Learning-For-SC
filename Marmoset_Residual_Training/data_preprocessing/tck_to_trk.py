import sys
from py_helpers.general_helpers import *
import numpy as np

import nibabel as nib

from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes

# Function to do the streamline node extraction
def streamline_node_extraction(tck_file, template, output_path):

    # Open dwi
    nii = nib.load(template)
    
    # Get the filename of the streamline
    streamline_filename = tck_file.split(os.sep)[-1].replace(".tck", ".trk")

    # Get the region folder name
    region_ID = tck_file.split(os.sep)[-3]
    region_folder = os.path.join(output_path, region_ID)
    check_output_folders(region_folder, "region_folder", wipe=False)

    # Define the new filepath
    new_filepath = os.path.join(region_folder, streamline_filename)

    # Load the tck
    tck = nib.streamlines.load(tck_file)

    # Make the header
    header = {}
    header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
    header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = nii.shape[:3]
    header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))

    # Save the new streamlines
    nib.streamlines.save(tck.tractogram, new_filepath, header=header)
    print("Saved new streamlines to {}".format(new_filepath))


# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    if hpc:
        data_path = "/rds/general/user/hsa22/ephemeral/Brain_MINDS/trk_data"
        template = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/model_data_w_resize/dMRI_b0/A10-R01_0028-TT21/DWI_concatenated_b0_resized.nii.gz"
        output_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/model_data_w_resize/trk_data_voxels"
    else:
        data_path = "/media/hsa22/Expansion/Brain-MINDS/BMCR_core_data/meta_data"
        template = "/media/hsa22/Expansion/Brain-MINDS/unused_model_data/dMRI_b0/A10-R01_0028-TT21/DWI_concatenated_b0.nii.gz"
        output_path = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/trk_data_originalsize"

    check_output_folders(output_path, "output_path", wipe=False)

    # Grab the tck files - should be 156 (3 types x 52 injections)
    tck = glob_files(data_path, "tck")

    # Get which region to run
    if hpc:
        file_idx = int(sys.argv[2])
        streamline_node_extraction(tck[file_idx], template, output_path)
    else:
        for file_idx in range(len(tck)):
            streamline_node_extraction(tck[file_idx], template, output_path)

if __name__ == "__main__":
    main()