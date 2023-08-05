import sys
from py_helpers.general_helpers import *
import numpy as np

import nibabel as nib

from nibabel.affines import apply_affine

import numpy.linalg as npl

# Function to do the streamline node extraction
def streamline_node_extraction(trk_file, template, output_path):
    
    # Get the filename of the streamline
    streamline_filename = trk_file.split(os.sep)[-1].replace(".trk", "_voxel_transform.trk")

    # Get the region folder name
    region_ID = trk_file.split(os.sep)[-2]
    region_folder = os.path.join(output_path, region_ID)
    check_output_folders(region_folder, "region_folder", wipe=False)

    # Define the new filepath
    new_filepath = os.path.join(region_folder, streamline_filename)

    # Load the streamlines and the template
    tractogram = nib.streamlines.load(trk_file)
    streamlines = tractogram.streamlines
    dwi = nib.load(template)

    # Get the affine matrix and invert
    affine = dwi.affine
    inv_affine = npl.inv(affine)

    new_streamlines = []

    for streamline in streamlines:
        new_streamlines.append(apply_affine(inv_affine, streamline))

    # Save the new streamlines
    nib.streamlines.save(nib.streamlines.Tractogram(new_streamlines, affine_to_rasmm=np.eye(4)), new_filepath, header=tractogram.header)
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
        data_path = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/resampled_streamlines"
        template = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/dMRI_b0/A10-R01_0028-TT21/DWI_concatenated_b0_resized.nii.gz"
        output_path = "/media/hsa22/Expansion/Brain-MINDS/model_data_w_resize/resampled_streamlines_voxels"

    check_output_folders(output_path, "output_path", wipe=False)

    # Grab the trk files - should be 156 (3 types x 52 injections)
    trk = glob_files(data_path, "trk")

    # Get which region to run
    if hpc:
        file_idx = int(sys.argv[2])
        streamline_node_extraction(trk[file_idx], template, output_path)
    else:
        for file_idx in range(len(trk)):
            streamline_node_extraction(trk[file_idx], template, output_path)

if __name__ == "__main__":
    main()