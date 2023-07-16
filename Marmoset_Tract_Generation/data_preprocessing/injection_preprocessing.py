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

    # --------------- TCKMAP all streamlines --------------- #
    # Define the folder to save the tckmapped streamlines and check the folder
    TCKMAPPED_STREAMLINES_FOLDER = os.path.join(BMINDS_DATA_FOLDER, "tckmapped_streamlines")
    TCKMAPPED_COMBINED_FOLDER = os.path.join(TCKMAPPED_STREAMLINES_FOLDER, "combined_tckmapped")
    check_output_folders(TCKMAPPED_STREAMLINES_FOLDER, "TCKMAPPED_STREAMLINES_FOLDER", wipe=False)
    check_output_folders(TCKMAPPED_COMBINED_FOLDER, "TCKMAPPED_COMBINED_FOLDER", wipe=False)

    # Define the path with the resized B0 files
    RESIZED_B0_FOLDER = os.path.join(BMINDS_DATA_FOLDER, "resized_B0")
    # Glob all the streamline files
    COMBINED_STREAMLINE_FILES = glob_files(RESIZED_B0_FOLDER, "tck")
    # Check the globbed files
    check_globbed_files(COMBINED_STREAMLINE_FILES, "COMBINED_STREAMLINE_FILES")

    # Get the streamline density file as a template
    STREAMLINE_DENSITY_FILE = [file for file in BMINDS_INJECTION_FILES if "streamline_density" in file][0]

    # For every file, make a flipped version
    for file in COMBINED_STREAMLINE_FILES:
        # First get the file name
        filename = file.split(os.sep)[-1] + ".nii.gz"
        # Define the path
        tckmapped_file_path = os.path.join(TCKMAPPED_COMBINED_FOLDER, filename)
        # Define the tckmap command
        TCKMAP_CMD = "tckmap {input} {output} -template {template} -force".format(input=file, 
                                                                           output=tckmapped_file_path, 
                                                                           template=STREAMLINE_DENSITY_FILE)
        # Run the command
        print("Running command: {}".format(TCKMAP_CMD))
        subprocess.run(TCKMAP_CMD, shell=True, check=True)

        # Get the flipped file name
        flipped_file = tckmapped_file_path.replace(".nii.gz", "_flipped.nii.gz")
        # Check if the flipped file exists
        if not os.path.isfile(flipped_file):
            # Flip the file
            flipped_image = flip_image(tckmapped_file_path)
            # Save the flipped file
            sitk.WriteImage(flipped_image, flipped_file)
    
    # Glob all nii.gz in the flipped streamlines folder
    TCKMAPPED_COMBINED_STREAMLINE_FILES = glob_files(TCKMAPPED_COMBINED_FOLDER, "nii.gz")
    # Check the globbed files
    check_globbed_files(TCKMAPPED_COMBINED_STREAMLINE_FILES, "TCKMAPPED_COMBINED_STREAMLINE_FILES")

    # Print the number of flipped streamline files
    print("Number of streamline files with flipping: {}".format(len(TCKMAPPED_COMBINED_STREAMLINE_FILES)))
    

if __name__ == "__main__":
    main()
