import os
import sys
sys.path.append("..")
from .inj_paths import *
from .inj_checkpoints import *
from .inj_general import *
from py_helpers.shared_helpers import *
import numpy as np

# Function to do all the mrtrix commands for each individual file rather than all
def mrtrix_all_region_functions(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    DWI_FILE = ARGS[1]
    STREAMLINE_FILE = ARGS[2]
    INJECTION_FILE = ARGS[3]
    ATLAS_STPT = ARGS[4]

    # Get the main paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_CONNECTOME_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, 
        INDIVIDUAL_ROIS_MIF_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Extract the ROIs of each atlas
    ATLAS_ROI_ARGS = [ATLAS_STPT, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME]
    (EXTRACTION_COMMANDS) = extract_each_roi_from_atlas(ATLAS_ROI_ARGS)

    # Convert each ROI to mif
    ROI_MIF_ARGS = [ATLAS_STPT]
    CONVERT_ROI_TO_MIF_CMD = convert_each_roi_to_mif(ROI_MIF_ARGS)

    # Define the injection mif commands
    INJECTION_MIF_ARGS = [REGION_ID, INJECTION_FILE]
    (CREATE_INJECTION_MIF_CMD) = create_mifs_of_each_injection_site(INJECTION_MIF_ARGS)

    # Define the tract editing commands - this will return a dictionary of commands
    TRACT_EDITING_ARGS = [REGION_ID, ATLAS_STPT, STREAMLINE_FILE]
    (FIND_STREAMLINES_CMD) = find_number_of_streamlines_between_injection_and_roi(TRACT_EDITING_ARGS)

    # Define the tract stats commands - this will return a dictionary of commands
    TRACT_STATS_ARGS = [REGION_ID, ATLAS_STPT]
    (FIND_STATS_CMD) = call_stats_between_injection_and_roi(TRACT_STATS_ARGS)

    # Check if we need to do the above commands
    CHECKPOINT_ARGS = [REGION_ID, ATLAS_STPT]
    (INJECTION_MIFS, MRTRIX_ATLAS_ROIS, MRTRIX_ATLAS_MIF_CONVERSION) = check_missing_region_files(CHECKPOINT_ARGS)

    # Create MRTRIX commands, depending on what we need to do
    MRTRIX_COMMANDS = []
    if INJECTION_MIFS:
        MRTRIX_COMMANDS.extend([
            (CREATE_INJECTION_MIF_CMD, "Creating injection mifs")
        ])
    if MRTRIX_ATLAS_ROIS:
        for idx, extraction in enumerate(EXTRACTION_COMMANDS):
            MRTRIX_COMMANDS.extend([
                (extraction, "Extracting ROI {} from the atlas".format(idx))
            ])
    if MRTRIX_ATLAS_MIF_CONVERSION:
        for idx, conversion in enumerate(CONVERT_ROI_TO_MIF_CMD):
            MRTRIX_COMMANDS.extend([
                (conversion, "Converting ROI {} to mif".format(idx))
            ])
    # For each type of command, add it to the MRTRIX_COMMANDS list
    for key, value in FIND_STREAMLINES_CMD.items():
        # For each command in the list, add it to the MRTRIX_COMMANDS list
        for idx, streamline_edit in enumerate(value):
            MRTRIX_COMMANDS.extend([
                (streamline_edit, "Finding streamlines between injection site and ROI {} using {}".format(idx, key))
            ])
    # For each type of command, add it to the MRTRIX_COMMANDS list
    for key, value in FIND_STATS_CMD.items():
        # For each command in the list, add it to the MRTRIX_COMMANDS list
        for idx, streamline_stats in enumerate(value):
            MRTRIX_COMMANDS.extend([
                (streamline_stats, "Finding stats of streamlines between injection site and ROI {} using {}".format(idx, key))
            ])
            
    # Return the commands
    return (MRTRIX_COMMANDS)

# Function to create mifs of each injection site
def create_mifs_of_each_injection_site(ARGS):
    
    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    INJECTION_FILE = ARGS[1]

    # Extract the cell density from the injection file
    TO_EXTRACT = ["cell_density"]
    CELL_DENSITY_FILE = extract_from_input_list(INJECTION_FILE, TO_EXTRACT, "injection")["cell_density"][0]

    # Get the injection mifs path
    INJECTION_MIF_PATH = get_injection_mif_path(REGION_ID)

    # Create the injection mif
    CREATE_INJECTION_MIF_CMD = "mrconvert {input} {output}.mif".format(input=CELL_DENSITY_FILE, output=INJECTION_MIF_PATH)

    # Return the command
    return (CREATE_INJECTION_MIF_CMD)

# Function to extract each ROI from the atlas
def extract_each_roi_from_atlas(ARGS):

    # Extract arguments needed to define paths
    ATLAS_STPT = ARGS[0]
    
    # Get the paths that we need
    (INDIVIDUAL_ROIS_FROM_ATLAS_PATH) = get_individual_rois_from_atlas_path(ATLAS_STPT)

    # This will hold all the extraction commands
    EXTRACTION_COMMANDS = []

    # Get the atlas and atlas labels path
    NEEDED_FILES_ATLAS = ["atlas_label"]
    ATLAS_LABEL_NEEDED_PATH = extract_from_input_list(ATLAS_STPT, NEEDED_FILES_ATLAS, "atlas_stpt")

    # Get the registered atlas path
    REG_ATLAS_PATH_IDX, REG_ATLAS_MIF_PATH_IDX = 0, 1
    REGISTERED_ATLAS_PATH = get_mrtrix_atlas_reg_paths_ants()[REG_ATLAS_PATH_IDX]

    # For every line in the atlas label file, extract the ROI
    with open(ATLAS_LABEL_NEEDED_PATH["atlas_label"], "r") as atlas_label_file:
        for line in atlas_label_file:
            # Get the ROI number and name - FROM THE ATLAS LABEL FILE
            LINE_SPLIT = [splits for splits in line.split("\t") if splits]
            ROI_NUM = LINE_SPLIT[0]
            ROI_NAME = LINE_SPLIT[-1].replace('"', '').replace(" ", "_").replace("\n", "").replace("(", "").replace(")", "")
            filename = "NUMBER_" + ROI_NUM + "_NAME_" + ROI_NAME
            # Get the atlas ROI path - FROM THE INDIVIDUAL ROIS FROM ATLAS PATH
            ATLAS_ROI_PATH = [file for file in INDIVIDUAL_ROIS_FROM_ATLAS_PATH if filename == file.split("/")[-1].split(".")[0]][0]
            # Extract the ROI from the atlas
            EXTRACT_ROI_CMD = "mrcalc {input}.nii.gz {roi_num} -eq {output}.nii.gz".format(input=REGISTERED_ATLAS_PATH,
                                                                                    roi_num=ROI_NUM, output=ATLAS_ROI_PATH) 
            # Add the command to the list
            EXTRACTION_COMMANDS.append(EXTRACT_ROI_CMD)
            
    # Return the commands
    return (EXTRACTION_COMMANDS)

# Function to convert each extracted ROI to mif
def convert_each_roi_to_mif(ARGS):

    # Extract arguments needed to define paths
    ATLAS_STPT = ARGS[0]

    # This will hold all the conversion commands
    CONVERSION_COMMANDS = []

    # Get the paths we need
    (INDIVIDUAL_ROIS_FROM_ATLAS_PATH) = get_individual_rois_from_atlas_path(ATLAS_STPT)
    # Get mif paths
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)
    
    # For every atlas ROI file, convert it to mif
    for nifti_roi_filepath in INDIVIDUAL_ROIS_FROM_ATLAS_PATH:
        # Get the output filename
        nifti_filename = nifti_roi_filepath.split("/")[-1]
        # Get the mif filename
        ROI_MIF_PATH = [mif_filepath for mif_filepath in INDIVIDUAL_ROIS_MIF_PATHS if nifti_filename == mif_filepath.split("/")[-1]][0]
        # Convert the atlas ROI to mif
        CONVERT_ROI_CMD = "mrconvert {input}.nii.gz {output}.mif".format(input=nifti_roi_filepath, output=ROI_MIF_PATH)
        # Add the command to the list
        CONVERSION_COMMANDS.append(CONVERT_ROI_CMD)

    # Return the commands
    return (CONVERSION_COMMANDS)

# Function to find the number of streamlines between each injection site and each ROI
def find_number_of_streamlines_between_injection_and_roi(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]
    STREAMLINE_FILE = ARGS[2]

    # This will hold all the commands
    TCKEDIT_COMMANDS = {}

    # This will define which TYPE we do, whether it's INCLUDES_BOTH, INCLUDES_ROI, or INCLUDES_ENDS_ONLY
    TYPES = ["includes_both", "includes_roi", "includes_ends_only"]

    # For each of the types, we need to find the number of streamlines between the injection site and the ROI
    for tckedit_type in TYPES:
        # Get the chosen tracts folder
        CHOSEN_TRACTS_FOLDER = get_chosen_tracts_stats_folder(REGION_ID, TYPE=tckedit_type, STATS=False)

        # This will tell us which ROIs are yet to be done
        (ROIS_TO_DO) = not_done_yet_injection_roi_tckedit(REGION_ID, ATLAS_STPT, CHOSEN_TRACTS_FOLDER, TYPE=tckedit_type)

        # Get the commands array and append it to the TCKEDIT_COMMANDS dictionary
        TCKEDIT_COMMANDS[tckedit_type] = get_tckedit_command(ATLAS_STPT, REGION_ID, ROIS_TO_DO, STREAMLINE_FILE, TYPE=tckedit_type)

    # Return the commands
    return (TCKEDIT_COMMANDS)

# Function to find the stats of the number of streamlines between each injection site and each ROI
def call_stats_between_injection_and_roi(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]

    # This will hold all the commands
    TCKSTATS_COMMANDS = {}

    # This will define which TYPE we do, whether it's INCLUDES_BOTH, INCLUDES_ROI, or INCLUDES_ENDS_ONLY
    TYPES = ["includes_both", "includes_roi", "includes_ends_only"]

    # For each of the types, we need to find the number of streamlines between the injection site and the ROI
    for tckstats_type in TYPES:
        # Get the chosen tracts folder
        CHOSEN_TRACTS_FOLDER = get_chosen_tracts_stats_folder(REGION_ID, TYPE=tckstats_type, STATS=True)

        # This will tell us which ROIs are yet to be done
        (ROIS_TO_DO) = not_done_yet_injection_roi_tckstats(REGION_ID, ATLAS_STPT, CHOSEN_TRACTS_FOLDER, TYPE=tckstats_type)

        # Get the commands array and append it to the TCKSTATS_COMMANDS dictionary
        TCKSTATS_COMMANDS[tckstats_type] = get_tckstats_command(ATLAS_STPT, REGION_ID, ROIS_TO_DO, TYPE=tckstats_type)

    # Return the commands
    return (TCKSTATS_COMMANDS)

# Function to actually grab the stats file and get various things
def grab_and_find_stats_between_injection_and_roi(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]

    # This will define which TYPE we do, whether it's INCLUDES_BOTH, INCLUDES_ROI, or INCLUDES_ENDS_ONLY
    TYPES = ["includes_both", "includes_roi", "includes_ends_only"]

    # For each of the types, find the stats results
    for stats_type in TYPES:
        find_and_save_stats_results(ATLAS_STPT, REGION_ID, TYPE=stats_type)

# Function to concatenate the results of the stats for ALL regions of each injection to make the connectome
def concatenate_all_roi_stats(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]

    # This will define which TYPE we do, whether it's INCLUDES_BOTH, INCLUDES_ROI, or INCLUDES_ENDS_ONLY
    TYPES = ["includes_both", "includes_roi", "includes_ends_only"]

    # For each of the types, concatenate the stats
    for stats_type in TYPES:
        # Get the chosen tracts stats folder
        CHOSEN_TRACTS_STATS_FOLDER = get_chosen_tracts_stats_folder(REGION_ID, TYPE=stats_type, STATS=True)
        # Get the injection matrices paths - these hold the stats
        (INJECTION_LENGTH_MATRIX_PATH, INJECTION_COUNT_MATRIX_PATH, INJECTION_MEAN_MATRIX_PATH, 
         INJECTION_MEDIAN_MATRIX_PATH, INJECTION_STD_MATRIX_PATH, INJECTION_MIN_MATRIX_PATH, 
         INJECTION_MAX_MATRIX_PATH) = get_injection_matrices_path(REGION_ID, TYPE=stats_type)
        # Get all the txt files in the stats folder
        STATS_FILES = glob_files(CHOSEN_TRACTS_STATS_FOLDER, "txt")
        # Filter files according to name
        LENGTH_FILES = [file for file in STATS_FILES if "length" in file]
        COUNT_FILES = [file for file in STATS_FILES if "count" in file]
        MEAN_FILES = [file for file in STATS_FILES if "mean" in file]
        MEDIAN_FILES = [file for file in STATS_FILES if "median" in file]
        STD_FILES = [file for file in STATS_FILES if "std" in file]
        MIN_FILES = [file for file in STATS_FILES if "min" in file]
        MAX_FILES = [file for file in STATS_FILES if "max" in file]
        # Make sure files aren't empty
        check_globbed_files(LENGTH_FILES, "LENGTH_FILES")
        check_globbed_files(COUNT_FILES, "COUNT_FILES")
        check_globbed_files(MEAN_FILES, "MEAN_FILES")
        check_globbed_files(MEDIAN_FILES, "MEDIAN_FILES")
        check_globbed_files(STD_FILES, "STD_FILES")
        check_globbed_files(MIN_FILES, "MIN_FILES")
        check_globbed_files(MAX_FILES, "MAX_FILES")
        # For every file, concatenate the stats into a list
        LENGTHS_DATA = read_stats_file(LENGTH_FILES)
        COUNT_DATA = read_stats_file(COUNT_FILES)
        MEAN_DATA = read_stats_file(MEAN_FILES)
        MEDIAN_DATA = read_stats_file(MEDIAN_FILES)
        STD_DATA = read_stats_file(STD_FILES)
        MIN_DATA = read_stats_file(MIN_FILES)
        MAX_DATA = read_stats_file(MAX_FILES)
        # Save the data to the stats files
        np.savetxt(INJECTION_LENGTH_MATRIX_PATH, LENGTHS_DATA, delimiter=",")
        np.savetxt(INJECTION_COUNT_MATRIX_PATH, COUNT_DATA, delimiter=",")
        np.savetxt(INJECTION_MEAN_MATRIX_PATH, MEAN_DATA, delimiter=",")
        np.savetxt(INJECTION_MEDIAN_MATRIX_PATH, MEDIAN_DATA, delimiter=",")
        np.savetxt(INJECTION_STD_MATRIX_PATH, STD_DATA, delimiter=",")
        np.savetxt(INJECTION_MIN_MATRIX_PATH, MIN_DATA, delimiter=",")
        np.savetxt(INJECTION_MAX_MATRIX_PATH, MAX_DATA, delimiter=",")

# Function to move all the existing data in injection_ROI_tracts to includes_both
def move_existing_data_to_includes_both(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]

    # Get the paths we need
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_TRACTS_FOLDER, INJECTION_ROI_TRACTS_STATS_FOLDER, 
     INJECTION_CONNECTOME_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_ROI_ONLY_FOLDER, 
     INJECTION_ROI_TRACTS_INCLUDES_ENDS_ONLY_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_BOTH_FOLDER, 
     INJECTION_ROI_TRACTS_STATS_INCLUDES_ROI_ONLY_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_ENDS_ONLY_FOLDER,
     INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, 
     INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER) = region_mrtrix_folder_paths(REGION_ID)
    
    # Grab all tck and txt files in the injection_ROI_tracts folder
    INJECTION_ROI_TRACTS_TCK_FILES = glob_files(INJECTION_ROI_TRACTS_FOLDER, "tck")
    INJECTION_ROI_TRACTS_TXT_FILES = glob_files(INJECTION_ROI_TRACTS_FOLDER, "txt")

    # Move all the tck and txt files to the includes_both folder
    if not os.path.exists(INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER):
        os.makedirs(INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER, exist_ok=True)
    
    # Move all the tck files
    for tck_file in INJECTION_ROI_TRACTS_TCK_FILES:
        shutil.move(tck_file, INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER)
    # Move all the txt files
    for txt_file in INJECTION_ROI_TRACTS_TXT_FILES:
        shutil.move(txt_file, INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER)