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

    # Define the tract editing commands
    TRACT_EDITING_ARGS = [REGION_ID, ATLAS_STPT]
    (FIND_STREAMLINES_CMD) = find_number_of_streamlines_between_injection_and_roi(TRACT_EDITING_ARGS)

    # Define the tract stats commands
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
    for idx, streamline_editing in enumerate(FIND_STREAMLINES_CMD):
        MRTRIX_COMMANDS.extend([
            (streamline_editing, "Finding streamlines between injection site and ROI {}".format(idx))
        ])
    for idx, streamline_stats in enumerate(FIND_STATS_CMD):
        MRTRIX_COMMANDS.extend([
            (streamline_stats, "Finding stats of streamlines between injection site and ROI {}".format(idx))
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

# DIDNT WORK DONT KNOW HOW TO MAKE MY OWN ATLAS INSTEAD JUST FIND THE TCKEDIT BETWEEN EACH INJECTION SITE AND EACH ROI
# AND THEN USE TCKSTATS TO FIND THE NUMBER OF STREAMLINES BETWEEN EACH INJECTION SITE AND EACH ROI USING COUNT AND THAT
# MAKES THE CONNECTOME

# Function to find the number of streamlines between each injection site and each ROI
def find_number_of_streamlines_between_injection_and_roi(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]

    # Get the paths we need
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_TRACTS_FOLDER, INJECTION_ROI_TRACTS_STATS_FOLDER,
    INJECTION_STATS_MATRIX_FOLDER) = region_mrtrix_folder_paths(REGION_ID)
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)
    (INJECTION_MIF_PATH) = get_injection_mif_path(REGION_ID)
    (COMBINED_TRACTS_PATH) = get_combined_tracts_path()
    (ATLAS_REG_PATH, ATLAS_REG_MIF_PATH) = get_mrtrix_atlas_reg_paths_ants()

    # This will hold all the commands
    TCKEDIT_COMMANDS = []

    # This will tell us which ROIs are yet to be done
    (ROIS_TO_DO) = not_done_yet_injection_roi_tckedit(REGION_ID, ATLAS_STPT, INJECTION_ROI_TRACTS_FOLDER)

    print("For region {}, we have {} ROIs to do".format(REGION_ID, len(ROIS_TO_DO)))

    # For every ROI, find the number of streamlines between the injection site and the ROI
    for idx, roi_mif_path in enumerate(INDIVIDUAL_ROIS_MIF_PATHS):
        # Get the ROI name or ID
        roi_name = roi_mif_path.split("/")[-1]
        # Check if we need to do this ROI
        if roi_name not in ROIS_TO_DO:
            print("Skipping ROI {}".format(roi_name))
            continue
        # Get the injection ROI tracts path
        (INJECTION_ROI_TRACTS_PATH) = get_injection_roi_tracts_path(REGION_ID, roi_name)
        # Find the number of streamlines between the injection site and the ROI
        FIND_STREAMLINES_CMD = "tckedit {all_tracts}.tck -include {inj_site}.mif -include {atlas_roi}.mif {output}.tck -force".format(
            all_tracts=COMBINED_TRACTS_PATH, inj_site=INJECTION_MIF_PATH, atlas_roi=ATLAS_REG_MIF_PATH, output=INJECTION_ROI_TRACTS_PATH)
        # Add the command to the list
        TCKEDIT_COMMANDS.append(FIND_STREAMLINES_CMD)

    # Return the commands
    return (TCKEDIT_COMMANDS)

# Function to find the stats of the number of streamlines between each injection site and each ROI
def call_stats_between_injection_and_roi(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]

    # Get the paths we need
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_TRACTS_FOLDER, INJECTION_ROI_TRACTS_STATS_FOLDER,
        INJECTION_STATS_MATRIX_FOLDER) = region_mrtrix_folder_paths(REGION_ID)
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)

    # This will hold all the commands
    TCKSTATS_COMMANDS = []

    # This will tell us which ROIs are yet to be done
    (ROIS_TO_DO) = not_done_yet_injection_roi_stats(REGION_ID, ATLAS_STPT, INJECTION_ROI_TRACTS_STATS_FOLDER)

    # For every ROI, find the stats of the number of streamlines between the injection site and the ROI
    for idx, roi_mif_path in enumerate(INDIVIDUAL_ROIS_MIF_PATHS):
        # Get the ROI name or ID
        roi_name = roi_mif_path.split("/")[-1]
        # Check if we need to do this ROI
        if roi_name not in ROIS_TO_DO:
            continue
        # Get the injection ROI tracts and stats path
        (INJECTION_ROI_TRACTS_PATH) = get_injection_roi_tracts_path(REGION_ID, roi_name)
        (INJECTION_ROI_LENGTHS_PATH, INJECTION_ROI_COUNT_PATH, INJECTION_ROI_MEAN_PATH, 
         INJECTION_ROI_MEDIAN_PATH, INJECTION_ROI_STD_PATH, INJECTION_ROI_MIN_PATH, 
         INJECTION_ROI_MAX_PATH) = get_injection_roi_tracts_stats_path(REGION_ID, roi_name)
        # Find the stats of the number of streamlines between the injection site and the ROI. Note that it PRINTS out
        # everything, but we can grab the count by counting the number of lines in the file
        # Can also get the mean, median, min, max, std, etc. by doing other modifications on the text file
        FIND_STATS_CMD = "tckstats {input}.tck -dump {output} -force".format(input=INJECTION_ROI_TRACTS_PATH, 
                                                                        output=INJECTION_ROI_LENGTHS_PATH)
        # Add the command to the list
        TCKSTATS_COMMANDS.append(FIND_STATS_CMD)

    # Return the commands
    return (TCKSTATS_COMMANDS)

# Function to actually grab the stats file and get various things
def grab_and_find_stats_between_injection_and_roi(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]

    # Get the paths we need
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)

    # For every ROI, find the stats of the number of streamlines between the injection site and the ROI
    for idx, roi_mif_path in enumerate(INDIVIDUAL_ROIS_MIF_PATHS):
        # Get the ROI name or ID
        roi_name = roi_mif_path.split("/")[-1]
        # Get the injection ROI stats path
        (INJECTION_ROI_LENGTHS_PATH, INJECTION_ROI_COUNT_PATH, INJECTION_ROI_MEAN_PATH, 
         INJECTION_ROI_MEDIAN_PATH, INJECTION_ROI_STD_PATH, INJECTION_ROI_MIN_PATH, 
         INJECTION_ROI_MAX_PATH) = get_injection_roi_tracts_stats_path(REGION_ID, roi_name)
        # 1- Get the number of lines in the text file of LENGTHS, which is the COUNT
        # 2- Get the mean, median, min, max, std, etc. by doing other modifications on the text file
        with open(INJECTION_ROI_LENGTHS_PATH, "r") as lengths_file:
            # Get the number of lines in the file
            count_data = len(lengths_file.readlines())
            lengths_data = [float(ln.rstrip()) for ln in lengths_file.readlines()] 
            mean_data = float(sum(lengths_data))/len(lengths_data) if len(lengths_data) > 0 else float('nan')
            median_data = float(sorted(lengths_data)[len(lengths_data)//2]) if len(lengths_data) > 0 else float('nan')
            min_data = float(min(lengths_data)) if len(lengths_data) > 0 else float('nan')
            max_data = float(max(lengths_data)) if len(lengths_data) > 0 else float('nan')
            std_data = np.std(np.array(lengths_data)) if len(lengths_data) > 0 else float('nan')

            # Save the count, mean, median, min, max, std, etc. to the text file
            with open(INJECTION_ROI_COUNT_PATH, "w") as count_file:
                count_file.write(str(count_data))
            with open(INJECTION_ROI_MEAN_PATH, "w") as mean_file:
                mean_file.write(str(mean_data))
            with open(INJECTION_ROI_MEDIAN_PATH, "w") as median_file:
                median_file.write(str(median_data))
            with open(INJECTION_ROI_MIN_PATH, "w") as min_file:
                min_file.write(str(min_data))
            with open(INJECTION_ROI_MAX_PATH, "w") as max_file:
                max_file.write(str(max_data))
            with open(INJECTION_ROI_STD_PATH, "w") as std_file:
                std_file.write(str(std_data))

# Function to concatenate the results of the stats for ALL regions of each injection to make the connectome
def concatenate_all_roi_stats(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]

    # Get the paths we need
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_TRACTS_FOLDER, INJECTION_ROI_TRACTS_STATS_FOLDER,
    INJECTION_STATS_MATRIX_FOLDER) = region_mrtrix_folder_paths(REGION_ID)
    (INJECTION_LENGTH_MATRIX_PATH, INJECTION_COUNT_MATRIX_PATH, INJECTION_MEAN_MATRIX_PATH, INJECTION_MEDIAN_MATRIX_PATH,
    INJECTION_STD_MATRIX_PATH, INJECTION_MIN_MATRIX_PATH, INJECTION_MAX_MATRIX_PATH) = get_injection_matrices_path(REGION_ID)

    # Get all the txt files in the stats folder
    STATS_FILES = glob_files(INJECTION_ROI_TRACTS_STATS_FOLDER, "txt")

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
    save_stats_vectors(LENGTHS_DATA, INJECTION_LENGTH_MATRIX_PATH)
    save_stats_vectors(COUNT_DATA, INJECTION_COUNT_MATRIX_PATH)
    save_stats_vectors(MEAN_DATA, INJECTION_MEAN_MATRIX_PATH)
    save_stats_vectors(MEDIAN_DATA, INJECTION_MEDIAN_MATRIX_PATH)
    save_stats_vectors(STD_DATA, INJECTION_STD_MATRIX_PATH)
    save_stats_vectors(MIN_DATA, INJECTION_MIN_MATRIX_PATH)
    save_stats_vectors(MAX_DATA, INJECTION_MAX_MATRIX_PATH)





