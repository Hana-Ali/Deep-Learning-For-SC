import sys
from py_helpers.general_helpers import *
import numpy as np

import SimpleITK as sitk

import subprocess

# Flip the image
def flip_image(image):

    # Get the image array
    image = sitk.ReadImage(image)
    image_array = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))

    # Get the spacing, direction and origin of the image
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()

    # Flip the image array
    image_array = np.flipud(image_array)

    # Get the new image
    new_image = sitk.GetImageFromArray(np.transpose(image_array, (2, 1, 0)))

    # Set the new spacing, direction and origin
    new_image.SetSpacing(spacing)
    new_image.SetDirection(direction)
    new_image.SetOrigin(origin)

    # Return the new image
    return new_image


# Main function
def main():
    # --------------- Get main folder paths, check inputs/outputs, unzip necessary --------------- #
    # Get whether it's HPC or not
    hpc = int(sys.argv[1])
    (BMINDS_DATA_FOLDER, BMINDS_OUTPUTS_DMRI_FOLDER, BMINDS_OUTPUTS_INJECTIONS_FOLDER, BMINDS_CORE_FOLDER,
    BMINDS_DWI_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_TEMPLATES_FOLDER, BMINDS_ATLAS_FOLDER, BMINDS_STPT_TEMPLATE_FOLDER, 
    BMINDS_TRANSFORMS_FOLDER, BMINDS_INJECTIONS_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER, BMINDS_UNZIPPED_DWI_RESIZED_FOLDER,
    MAIN_MRTRIX_FOLDER_DMRI, MAIN_MRTRIX_FOLDER_INJECTIONS) = get_main_paths(hpc)
    
    # Check the input folders
    check_input_folders(BMINDS_DATA_FOLDER, "BMINDS_DATA_FOLDER")
    check_input_folders(BMINDS_CORE_FOLDER, "BMINDS_CORE_FOLDER")
    check_input_folders(BMINDS_DWI_FOLDER, "BMINDS_DWI_FOLDER")
    check_input_folders(BMINDS_METADATA_FOLDER, "BMINDS_METADATA_FOLDER")
    check_input_folders(BMINDS_TEMPLATES_FOLDER, "BMINDS_TEMPLATES_FOLDER")
    check_input_folders(BMINDS_ATLAS_FOLDER, "BMINDS_ATLAS_FOLDER")
    check_input_folders(BMINDS_STPT_TEMPLATE_FOLDER, "BMINDS_STPT_TEMPLATE_FOLDER")
    check_input_folders(BMINDS_TRANSFORMS_FOLDER, "BMINDS_TRANSFORMS_FOLDER")
    check_input_folders(BMINDS_INJECTIONS_FOLDER, "BMINDS_INJECTIONS_FOLDER")

    # Check the output folders
    check_output_folders(BMINDS_OUTPUTS_INJECTIONS_FOLDER, "BMINDS_OUTPUTS_FOLDER", wipe=False)
    check_output_folders(BMINDS_UNZIPPED_DWI_FOLDER, "BMINDS_UNZIPPED_DWI_FOLDER", wipe=False)
    check_output_folders(MAIN_MRTRIX_FOLDER_INJECTIONS, "MAIN_MRTRIX_FOLDER_INJECTIONS", wipe=False)

    # --------------- Get the injection files --------------- #
    BMINDS_UNZIPPED_DWI_FILES = glob_files(BMINDS_UNZIPPED_DWI_FOLDER, "nii")
    BMINDS_BVAL_FILES = glob_files(BMINDS_DWI_FOLDER, "bval")
    BMINDS_BVEC_FILES = glob_files(BMINDS_DWI_FOLDER, "bvec")
    BMINDS_STREAMLINE_FILES = glob_files(BMINDS_METADATA_FOLDER, "tck")
    BMINDS_INJECTION_FILES = glob_files(BMINDS_INJECTIONS_FOLDER, "nii.gz")
    BMINDS_ATLAS_FILES = glob_files(BMINDS_ATLAS_FOLDER, "nii.gz")
    BMINDS_ATLAS_LABEL_FILES = glob_files(BMINDS_ATLAS_FOLDER, "txt")
    BMINDS_STPT_FILES = glob_files(BMINDS_STPT_TEMPLATE_FOLDER, "nii")
    BMINDS_TRANSFORM_FILES = glob_files(BMINDS_TRANSFORMS_FOLDER, "h5")

    # Get the atlas and stpt files - separate from the mix above
    BMINDS_ATLAS_FILE = [file for file in BMINDS_ATLAS_FILES if "140_region_atlas_segmentation" in file]
    BMINDS_ATLAS_LABEL_FILE = [file for file in BMINDS_ATLAS_LABEL_FILES if "140_region_atlas_labels" in file]
    BMINDS_STPT_FILE = [file for file in BMINDS_STPT_FILES if "STPT_template_unzipped" in file]
    BMINDS_MBCA_TRANSFORM_FILE = [file for file in BMINDS_TRANSFORM_FILES if "MBCA" in file]

    # Check the globbed files
    check_globbed_files(BMINDS_UNZIPPED_DWI_FILES, "BMINDS_UNZIPPED_DWI_FILES")
    check_globbed_files(BMINDS_BVAL_FILES, "BMINDS_BVAL_FILES")
    check_globbed_files(BMINDS_BVEC_FILES, "BMINDS_BVEC_FILES")
    check_globbed_files(BMINDS_STREAMLINE_FILES, "BMINDS_STREAMLINE_FILES")
    check_globbed_files(BMINDS_INJECTION_FILES, "BMINDS_INJECTION_FILES")
    check_globbed_files(BMINDS_ATLAS_FILE, "BMINDS_ATLAS_FILE")
    check_globbed_files(BMINDS_ATLAS_LABEL_FILE, "BMINDS_ATLAS_LABEL_FILE")
    check_globbed_files(BMINDS_STPT_FILE, "BMINDS_STPT_FILE")
    check_globbed_files(BMINDS_MBCA_TRANSFORM_FILE, "BMINDS_MBCA_TRANSFORM_FILE")

    # --------------- Extract B0 from all the resized files --------------- #
    # Get the resized files from the unzipped folder
    unzipped_nii_files = glob_files(BMINDS_UNZIPPED_DWI_FOLDER, "nii")
    unzipped_bval_files = glob_files(BMINDS_UNZIPPED_DWI_FOLDER, "bvals")
    unzipped_bvec_files = glob_files(BMINDS_UNZIPPED_DWI_FOLDER, "bvecs")

    # Get the resized files from the ones above
    resized_nii_files = [file for file in unzipped_nii_files if "concatenated_resized" in file]
    resized_bval_files = [file for file in unzipped_bval_files if "concatenated_resized" in file]
    resized_bvec_files = [file for file in unzipped_bvec_files if "concatenated_resized" in file]

    print("Length of unzipped nii files: {}".format(len(unzipped_nii_files)))
    print("Length of unzipped bval files: {}".format(len(unzipped_bval_files)))
    print("Length of unzipped bvec files: {}".format(len(unzipped_bvec_files)))

    print("Resized nii files: {}".format(resized_nii_files))
    print("Resized bval files: {}".format(resized_bval_files))
    print("Resized bvec files: {}".format(resized_bvec_files))

    # For each file
    for i in range(len(resized_nii_files)):
        # Get the region_ID
        region_ID = resized_nii_files[i].split(os.sep)[-3]
        # Create the new folder in the resized unzipped folder if it doesn't exist
        new_folder = os.path.join(BMINDS_UNZIPPED_DWI_RESIZED_FOLDER, region_ID, resized_nii_files[i].split(os.sep)[-2])
        check_output_folders(new_folder, "resized region folder", wipe=False)
        # Get the new filepath
        new_nii_filepath = os.path.join(new_folder, resized_nii_files[i].split(os.sep)[-1])
        new_bval_filepath = os.path.join(new_folder, resized_bval_files[i].split(os.sep)[-1])
        new_bvec_filepath = os.path.join(new_folder, resized_bvec_files[i].split(os.sep)[-1])
        print("New nii filepath: {}".format(new_nii_filepath))
        print("New bval filepath: {}".format(new_bval_filepath))
        print("New bvec filepath: {}".format(new_bvec_filepath))
        # Copy the files to the new folder
        shutil.copyfile(resized_nii_files[i], new_nii_filepath)
        shutil.copyfile(resized_bval_files[i], new_bval_filepath)
        shutil.copyfile(resized_bvec_files[i], new_bvec_filepath)

    # Get the resized files from the unzipped folder
    resized_mif_files = glob_files(BMINDS_UNZIPPED_DWI_RESIZED_FOLDER, "mif")
    resized_nii_gz_files = glob_files(BMINDS_UNZIPPED_DWI_RESIZED_FOLDER, "nii.gz")

    # Get only the ones with b0
    resized_mif_files = [file for file in resized_mif_files if "b0" in file]
    resized_nii_gz_files = [file for file in resized_nii_gz_files if "b0" in file]

    print("Length of resized mif files: {}".format(len(resized_mif_files)))
    print("Length of resized nii.gz files: {}".format(len(resized_nii_gz_files)))

    # For every file
    for file in resized_mif_files:
        # Define the output file
        new_path = ("").join(file.split(".mif")[:-1]) + ".mif"
        # Rename the file
        os.rename(file, new_path)
    
    # For every file
    for file in resized_nii_gz_files:
        # Define the output file
        new_path = ("").join(file.split(".nii.gz")[:-1]) + ".nii.gz"
        # Rename the file
        os.rename(file, new_path)

if __name__ == "__main__":
    main()
