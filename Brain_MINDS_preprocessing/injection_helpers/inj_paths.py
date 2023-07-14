import os
import sys
sys.path.append("..")
from py_helpers.general_helpers import *
from py_helpers.shared_helpers import *


# Define the MRTRIX folder paths
def main_mrtrix_folder_paths():

    # Get the general paths
    (BMINDS_DATA_FOLDER, BMINDS_OUTPUTS_DMRI_FOLDER, BMINDS_OUTPUTS_INJECTIONS_FOLDER, BMINDS_CORE_FOLDER,
    BMINDS_DWI_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_TEMPLATES_FOLDER, BMINDS_ATLAS_FOLDER, BMINDS_STPT_TEMPLATE_FOLDER, 
    BMINDS_TRANSFORMS_FOLDER, BMINDS_INJECTIONS_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER, BMINDS_UNZIPPED_DWI_RESIZED_FOLDER,
    MAIN_MRTRIX_FOLDER_DMRI, MAIN_MRTRIX_FOLDER_INJECTIONS) = get_main_paths(hpc=True)

    # Create the region's main folder - don't wipe as it'll be invoked more than once
    GENERAL_MRTRIX_FOLDER = os.path.join(MAIN_MRTRIX_FOLDER_INJECTIONS, "general_mrtrix")
    SPECIFIC_MRTRIX_FOLDER = os.path.join(MAIN_MRTRIX_FOLDER_INJECTIONS, "specific_mrtrix")
    ATLAS_REG_FOLDER_NAME = os.path.join(GENERAL_MRTRIX_FOLDER, "atlas_reg")
    COMBINED_TRACTS_FOLDER_NAME = os.path.join(GENERAL_MRTRIX_FOLDER, "combined_tracts")
    COMBINED_CONNECTOME_FOLDER_NAME = os.path.join(GENERAL_MRTRIX_FOLDER, "combined_connectome")
    DENSITY_MAPS_FOLDER_NAME = os.path.join(GENERAL_MRTRIX_FOLDER, "density_maps")

    INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME = os.path.join(GENERAL_MRTRIX_FOLDER, "individual_rois_from_atlas")
    INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME = os.path.join(INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, "nifti")
    INDIVIDUAL_ROIS_MIF_FOLDER_NAME = os.path.join(INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, "mif")
    check_output_folders(GENERAL_MRTRIX_FOLDER, "MRtrix general folder", wipe=False)
    check_output_folders(SPECIFIC_MRTRIX_FOLDER, "MRtrix specific folder", wipe=False)
    check_output_folders(ATLAS_REG_FOLDER_NAME, "MRtrix atlas reg folder", wipe=False)
    check_output_folders(COMBINED_TRACTS_FOLDER_NAME, "MRtrix combined tracts folder", wipe=False)
    check_output_folders(COMBINED_CONNECTOME_FOLDER_NAME, "MRtrix combined connectome folder", wipe=False)
    check_output_folders(DENSITY_MAPS_FOLDER_NAME, "MRtrix density maps folder", wipe=False)

    check_output_folders(INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, "MRtrix individual rois from atlas folder", wipe=False)
    check_output_folders(INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, "MRtrix individual rois from atlas nifti folder", wipe=False)
    check_output_folders(INDIVIDUAL_ROIS_MIF_FOLDER_NAME, "MRtrix individual rois from atlas to mif folder", wipe=False)


    # Return the paths
    return (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
            COMBINED_CONNECTOME_FOLDER_NAME, DENSITY_MAPS_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, 
            INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME)

# Define MRTRIX region-specific folder paths
def region_mrtrix_folder_paths(REGION_ID):
    
    # Get the main paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
    COMBINED_CONNECTOME_FOLDER_NAME, DENSITY_MAPS_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, 
    INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME) = main_mrtrix_folder_paths()
    
    # Create the region folder
    REGION_MRTRIX_FOLDER = os.path.join(SPECIFIC_MRTRIX_FOLDER, REGION_ID)
    INJECTION_MIF_FOLDER = os.path.join(REGION_MRTRIX_FOLDER, "injection_mifs")
    INJECTION_ROI_TRACTS_FOLDER = os.path.join(REGION_MRTRIX_FOLDER, "injection_ROI_tracts")
    INJECTION_ROI_TRACTS_STATS_FOLDER = os.path.join(REGION_MRTRIX_FOLDER, "injection_ROI_tracts_stats")
    INJECTION_CONNECTOME_FOLDER = os.path.join(REGION_MRTRIX_FOLDER, "injection_connectome")
    # We have a different roi tracts folder depending on whether we include both, include one, or use ends only
    INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER = os.path.join(INJECTION_ROI_TRACTS_FOLDER, "includes_both")
    INJECTION_ROI_TRACTS_INCLUDES_ROI_ONLY_FOLDER = os.path.join(INJECTION_ROI_TRACTS_FOLDER, "includes_roi")
    INJECTION_ROI_TRACTS_INCLUDES_ENDS_ONLY_FOLDER = os.path.join(INJECTION_ROI_TRACTS_FOLDER, "includes_ends_only")
    INJECTION_ROI_TRACTS_STATS_INCLUDES_BOTH_FOLDER = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER, "includes_both")
    INJECTION_ROI_TRACTS_STATS_INCLUDES_ROI_ONLY_FOLDER = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER, "includes_roi")
    INJECTION_ROI_TRACTS_STATS_INCLUDES_ENDS_ONLY_FOLDER = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER, "includes_ends_only")
    # We have different connectomes depending on whether we include both, include one, or use ends only in the previous step
    INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER = os.path.join(INJECTION_CONNECTOME_FOLDER, "includes_both")
    INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER = os.path.join(INJECTION_CONNECTOME_FOLDER, "includes_roi")
    INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER = os.path.join(INJECTION_CONNECTOME_FOLDER, "includes_ends_only")
    check_output_folders(REGION_MRTRIX_FOLDER, "REGION MRTRIX PATH", wipe=False)
    check_output_folders(INJECTION_MIF_FOLDER, "INJECTION MIF PATH", wipe=False)
    check_output_folders(INJECTION_ROI_TRACTS_FOLDER, "INJECTION ROI TRACTS PATH", wipe=False)
    check_output_folders(INJECTION_ROI_TRACTS_STATS_FOLDER, "INJECTION ROI TRACTS STATS PATH", wipe=False)
    check_output_folders(INJECTION_CONNECTOME_FOLDER, "INJECTION CONNECTOME PATH", wipe=False)
    check_output_folders(INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER, "INJECTION ROI TRACTS INCLUDES BOTH PATH", wipe=False)
    check_output_folders(INJECTION_ROI_TRACTS_INCLUDES_ROI_ONLY_FOLDER, "INJECTION ROI TRACTS INCLUDES ROI ONLY PATH", wipe=False)
    check_output_folders(INJECTION_ROI_TRACTS_INCLUDES_ENDS_ONLY_FOLDER, "INJECTION ROI TRACTS INCLUDES ENDS ONLY PATH", wipe=False)
    check_output_folders(INJECTION_ROI_TRACTS_STATS_INCLUDES_BOTH_FOLDER, "INJECTION ROI TRACTS STATS INCLUDES BOTH PATH", wipe=False)
    check_output_folders(INJECTION_ROI_TRACTS_STATS_INCLUDES_ROI_ONLY_FOLDER, "INJECTION ROI TRACTS STATS INCLUDES ROI ONLY PATH", wipe=False)
    check_output_folders(INJECTION_ROI_TRACTS_STATS_INCLUDES_ENDS_ONLY_FOLDER, "INJECTION ROI TRACTS STATS INCLUDES ENDS ONLY PATH", wipe=False)
    check_output_folders(INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, "INJECTION CONNECTOME INCLUDES BOTH PATH", wipe=False)
    check_output_folders(INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, "INJECTION CONNECTOME INCLUDES ROI ONLY PATH", wipe=False)
    check_output_folders(INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER, "INJECTION CONNECTOME INCLUDES ENDS ONLY PATH", wipe=False)

    # Return the paths
    return (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_TRACTS_FOLDER, INJECTION_ROI_TRACTS_STATS_FOLDER, INJECTION_CONNECTOME_FOLDER,
            INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_ROI_ONLY_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_ENDS_ONLY_FOLDER,
            INJECTION_ROI_TRACTS_STATS_INCLUDES_BOTH_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_ROI_ONLY_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_ENDS_ONLY_FOLDER,
            INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER)

# Function to define the atlas registration USING ANTS AND TRANSFORMATION MATRIX files
def get_mrtrix_atlas_reg_paths_ants():

    # Get the folder names
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
    COMBINED_CONNECTOME_FOLDER_NAME, DENSITY_MAPS_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, 
    INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Immediately transform with the transformation matrix
    ATLAS_REG_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "atlasreg_ANTS")
    ATLAS_REG_MIF_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "atlasreg_ANTS_mif")

    # Return the paths
    return (ATLAS_REG_PATH, ATLAS_REG_MIF_PATH)

# Function to define the combined tracts path
def get_combined_tracts_path():

    # Get the folder names and paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
    COMBINED_CONNECTOME_FOLDER_NAME, DENSITY_MAPS_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, 
    INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Define the combined tracts path
    COMBINED_TRACTS_PATH = os.path.join(COMBINED_TRACTS_FOLDER_NAME, "combined_tracts")

    # Return the path
    return (COMBINED_TRACTS_PATH)

# Function to define the injection mif path
def get_injection_mif_path(REGION_ID):
    
    # Get the folder names and paths
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_TRACTS_FOLDER, INJECTION_ROI_TRACTS_STATS_FOLDER, 
     INJECTION_CONNECTOME_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_ROI_ONLY_FOLDER, 
     INJECTION_ROI_TRACTS_INCLUDES_ENDS_ONLY_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_BOTH_FOLDER, 
     INJECTION_ROI_TRACTS_STATS_INCLUDES_ROI_ONLY_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_ENDS_ONLY_FOLDER,
     INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, 
     INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER) = region_mrtrix_folder_paths(REGION_ID)

    # Define the injection mif path
    INJECTION_MIF_PATH = os.path.join(INJECTION_MIF_FOLDER, "{}_injection".format(REGION_ID))

    # Return the path
    return (INJECTION_MIF_PATH)

# Function to define the individual rois from atlas path
def get_individual_rois_from_atlas_path(ATLAS_STPT):
    
    # Get the folder names and paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
    COMBINED_CONNECTOME_FOLDER_NAME, DENSITY_MAPS_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, 
    INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Get the atlas and atlas labels path
    NEEDED_FILES_ATLAS = ["atlas", "atlas_label"]
    ATLAS_LABEL_NEEDED_PATH = extract_from_input_list(ATLAS_STPT, NEEDED_FILES_ATLAS, "atlas_stpt")

    # This holds all the atlas ROI paths
    INDIVIDUAL_ROIS_FROM_ATLAS_PATH = []

    # For every line in the atlas label file, extract the ROI name
    with open(ATLAS_LABEL_NEEDED_PATH["atlas_label"], "r") as atlas_label_file:
        for line in atlas_label_file:
            # Get the ROI number and name
            LINE_SPLIT = [splits for splits in line.split("\t") if splits]
            ROI_NUM = LINE_SPLIT[0]
            ROI_NAME = LINE_SPLIT[-1].replace('"', '').replace(" ", "_").replace("\n", "").replace("(", "").replace(")", "")
            # Get the atlas ROI path
            filename = "NUMBER_" + ROI_NUM + "_NAME_" + ROI_NAME
            ATLAS_ROI_PATH = os.path.join(INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, filename)
            # Append the atlas ROI path to the list
            INDIVIDUAL_ROIS_FROM_ATLAS_PATH.append(ATLAS_ROI_PATH)

    # Return the path
    return (INDIVIDUAL_ROIS_FROM_ATLAS_PATH)

# Function to define the individual rois mif path
def get_individual_rois_mif_path(ATLAS_STPT):

    # Get the folder names and paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
    COMBINED_CONNECTOME_FOLDER_NAME, DENSITY_MAPS_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, 
    INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME) = main_mrtrix_folder_paths()
    
    # Grab all the ROI paths
    (INDIVIDUAL_ROIS_FROM_ATLAS_PATH) = get_individual_rois_from_atlas_path(ATLAS_STPT)

    # This holds all the atlas ROI mif paths
    INDIVIDUAL_ROIS_MIF_PATHS = []

    # For every nifti file, create the mif filename
    for nifti_file in INDIVIDUAL_ROIS_FROM_ATLAS_PATH:
        # Get the filename filename
        filename = nifti_file.split("/")[-1]
        # Create the mif path
        mif_path = os.path.join(INDIVIDUAL_ROIS_MIF_FOLDER_NAME, filename)
        # Append the mif path to the list
        INDIVIDUAL_ROIS_MIF_PATHS.append(mif_path)

    # Return the path
    return (INDIVIDUAL_ROIS_MIF_PATHS)

# Function to define the injection ROI tracts path
def get_injection_roi_tracts_path(REGION_ID, ROI_ID, TYPE="includes_both"):

    # Get the folder names and paths
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_TRACTS_FOLDER, INJECTION_ROI_TRACTS_STATS_FOLDER, 
     INJECTION_CONNECTOME_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_ROI_ONLY_FOLDER, 
     INJECTION_ROI_TRACTS_INCLUDES_ENDS_ONLY_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_BOTH_FOLDER, 
     INJECTION_ROI_TRACTS_STATS_INCLUDES_ROI_ONLY_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_ENDS_ONLY_FOLDER,
     INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, 
     INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER) = region_mrtrix_folder_paths(REGION_ID)

    # Define the injection ROI tracts path
    if TYPE == "includes_both":
        INJECTION_ROI_TRACTS_PATH = os.path.join(INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER, "ROI_{}_tracts".format(ROI_ID))
    elif TYPE == "includes_roi":
        INJECTION_ROI_TRACTS_PATH = os.path.join(INJECTION_ROI_TRACTS_INCLUDES_ROI_ONLY_FOLDER, "ROI_{}_tracts".format(ROI_ID))
    elif TYPE == "includes_ends_only":
        INJECTION_ROI_TRACTS_PATH = os.path.join(INJECTION_ROI_TRACTS_INCLUDES_ENDS_ONLY_FOLDER, "ROI_{}_tracts".format(ROI_ID))
    else:
        print("ERROR: Type {} not recognized".format(TYPE))
        sys.exit()

    # Return the path
    return (INJECTION_ROI_TRACTS_PATH)

# Function to define the injection ROI tracts stats path
def get_injection_roi_tracts_stats_path(REGION_ID, ROI_ID, TYPE="includes_both"):
    
    # Get the folder names and paths
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_TRACTS_FOLDER, INJECTION_ROI_TRACTS_STATS_FOLDER, 
     INJECTION_CONNECTOME_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_ROI_ONLY_FOLDER, 
     INJECTION_ROI_TRACTS_INCLUDES_ENDS_ONLY_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_BOTH_FOLDER, 
     INJECTION_ROI_TRACTS_STATS_INCLUDES_ROI_ONLY_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_ENDS_ONLY_FOLDER,
     INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, 
     INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER) = region_mrtrix_folder_paths(REGION_ID)
    # Make new folder per region
    INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC = os.path.join(INJECTION_ROI_TRACTS_STATS_INCLUDES_BOTH_FOLDER, ROI_ID)
    check_output_folders(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC", wipe=False)

    # Define the injection ROI tracts stats path, depending on the type
    if TYPE == "includes_both":
        INJECTION_ROI_LENGTHS_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_length.txt".format(ROI_ID))
        INJECTION_ROI_COUNT_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_count.txt".format(ROI_ID))
        INJECTION_ROI_MEAN_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_mean.txt".format(ROI_ID))
        INJECTION_ROI_MEDIAN_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_median.txt".format(ROI_ID))
        INJECTION_ROI_STD_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_std.txt".format(ROI_ID))
        INJECTION_ROI_MIN_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_min.txt".format(ROI_ID))
        INJECTION_ROI_MAX_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_max.txt".format(ROI_ID))
    elif TYPE == "includes_roi":
        INJECTION_ROI_LENGTHS_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_length.txt".format(ROI_ID))
        INJECTION_ROI_COUNT_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_count.txt".format(ROI_ID))
        INJECTION_ROI_MEAN_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_mean.txt".format(ROI_ID))
        INJECTION_ROI_MEDIAN_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_median.txt".format(ROI_ID))
        INJECTION_ROI_STD_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_std.txt".format(ROI_ID))
        INJECTION_ROI_MIN_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_min.txt".format(ROI_ID))
        INJECTION_ROI_MAX_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_max.txt".format(ROI_ID))
    elif TYPE == "includes_ends_only":
        INJECTION_ROI_LENGTHS_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_length.txt".format(ROI_ID))
        INJECTION_ROI_COUNT_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_count.txt".format(ROI_ID))
        INJECTION_ROI_MEAN_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_mean.txt".format(ROI_ID))
        INJECTION_ROI_MEDIAN_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_median.txt".format(ROI_ID))
        INJECTION_ROI_STD_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_std.txt".format(ROI_ID))
        INJECTION_ROI_MIN_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_min.txt".format(ROI_ID))
        INJECTION_ROI_MAX_PATH = os.path.join(INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, "ROI_{}_tracts_max.txt".format(ROI_ID))
    else:
        print("ERROR: Type {} not recognized".format(TYPE))
        sys.exit()

    # Return the path
    return (INJECTION_ROI_TRACTS_STATS_FOLDER_SPECIFIC, INJECTION_ROI_LENGTHS_PATH, INJECTION_ROI_COUNT_PATH, INJECTION_ROI_MEAN_PATH, INJECTION_ROI_MEDIAN_PATH,
            INJECTION_ROI_STD_PATH, INJECTION_ROI_MIN_PATH, INJECTION_ROI_MAX_PATH)

# Function to define the injection connectome path
def get_injection_matrices_path(REGION_ID, TYPE="includes_both"):

    # Get the folder names and paths
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_TRACTS_FOLDER, INJECTION_ROI_TRACTS_STATS_FOLDER, 
     INJECTION_CONNECTOME_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_ROI_ONLY_FOLDER, 
     INJECTION_ROI_TRACTS_INCLUDES_ENDS_ONLY_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_BOTH_FOLDER, 
     INJECTION_ROI_TRACTS_STATS_INCLUDES_ROI_ONLY_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_ENDS_ONLY_FOLDER,
     INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, 
     INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER) = region_mrtrix_folder_paths(REGION_ID)

    # Define the injection connectome path, depending on the type
    if TYPE == "includes_both":
        INJECTION_LENGTH_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, "{}_vector_length.txt".format(REGION_ID))
        INJECTION_COUNT_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, "{}_vector_count.txt".format(REGION_ID))
        INJECTION_MEAN_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, "{}_vector_mean.txt".format(REGION_ID))
        INJECTION_MEDIAN_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, "{}_vector_median.txt".format(REGION_ID))
        INJECTION_STD_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, "{}_vector_std.txt".format(REGION_ID))
        INJECTION_MIN_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, "{}_vector_min.txt".format(REGION_ID))
        INJECTION_MAX_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, "{}_vector_max.txt".format(REGION_ID))
    elif TYPE == "includes_roi":
        INJECTION_LENGTH_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, "{}_vector_length.txt".format(REGION_ID))
        INJECTION_COUNT_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, "{}_vector_count.txt".format(REGION_ID))
        INJECTION_MEAN_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, "{}_vector_mean.txt".format(REGION_ID))
        INJECTION_MEDIAN_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, "{}_vector_median.txt".format(REGION_ID))
        INJECTION_STD_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, "{}_vector_std.txt".format(REGION_ID))
        INJECTION_MIN_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, "{}_vector_min.txt".format(REGION_ID))
        INJECTION_MAX_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, "{}_vector_max.txt".format(REGION_ID))
    elif TYPE == "includes_ends_only":
        INJECTION_LENGTH_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER, "{}_vector_length.txt".format(REGION_ID))
        INJECTION_COUNT_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER, "{}_vector_count.txt".format(REGION_ID))
        INJECTION_MEAN_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER, "{}_vector_mean.txt".format(REGION_ID))
        INJECTION_MEDIAN_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER, "{}_vector_median.txt".format(REGION_ID))
        INJECTION_STD_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER, "{}_vector_std.txt".format(REGION_ID))
        INJECTION_MIN_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER, "{}_vector_min.txt".format(REGION_ID))
        INJECTION_MAX_MATRIX_PATH = os.path.join(INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER, "{}_vector_max.txt".format(REGION_ID))
    else:
        print("ERROR: Type {} not recognized".format(TYPE))
        sys.exit()

    # Return the path
    return (INJECTION_LENGTH_MATRIX_PATH, INJECTION_COUNT_MATRIX_PATH, INJECTION_MEAN_MATRIX_PATH, INJECTION_MEDIAN_MATRIX_PATH,
            INJECTION_STD_MATRIX_PATH, INJECTION_MIN_MATRIX_PATH, INJECTION_MAX_MATRIX_PATH)

# Function to define the final major connectome path
def get_combined_connectome_path():

    # Get the folder names and paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
    COMBINED_CONNECTOME_FOLDER_NAME, DENSITY_MAPS_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, 
    INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Define the combined connectome path
    COMBINED_CONNECTOME_PATH = os.path.join(COMBINED_CONNECTOME_FOLDER_NAME, "combined_connectome.csv")

    # Return the path
    return (COMBINED_CONNECTOME_PATH)

# Function to get the TDI path
def get_tdi_path(streamline_file):
    
    # Get the folder names and paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME,
    COMBINED_CONNECTOME_FOLDER_NAME, DENSITY_MAPS_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME,
    INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Get the region ID from the streamline file
    if os.name == "nt":
        REGION_ID = streamline_file.split("\\")[-3]
        FILE_NAME = streamline_file.split("\\")[-1]
    else:
        REGION_ID = streamline_file.split("/")[-3]
        FILE_NAME = streamline_file.split("/")[-1]

    # Create the region folder
    TDI_REGION_FOLDER = os.path.join(DENSITY_MAPS_FOLDER_NAME, "{}_TDI_files".format(REGION_ID))

    # Define the TDI path
    TDI_PATH = os.path.join(TDI_REGION_FOLDER, "{}_TDI".format(FILE_NAME))

    # Return the path
    return (TDI_REGION_FOLDER, TDI_PATH)