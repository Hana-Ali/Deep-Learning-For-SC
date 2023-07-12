import os

import sys
sys.path.append("..")
from py_helpers.general_helpers import *
from py_helpers.shared_helpers import *
from .inj_general_commands import *
from numpy import random
import numpy as np

# Create a list that associates each subject with its T1 and DWI files
def create_data_list(BMINDS_UNZIPPED_DWI_FILES, BMINDS_BVAL_FILES, BMINDS_BVEC_FILES, BMINDS_STREAMLINE_FILES, 
                     BMINDS_INJECTION_FILES, BMINDS_ATLAS_FILE, BMINDS_ATLAS_LABEL_FILE, BMINDS_STPT_FILE, 
                     BMINDS_MBCA_TRANSFORM_FILE):
    
    # This will hold all of the data lists
    DATA_LISTS = []
    
    # Get the initial lists
    (DWI_LIST, STREAMLINE_LIST, INJECTION_LIST) = create_initial_lists(BMINDS_UNZIPPED_DWI_FILES, BMINDS_BVAL_FILES,
                                                                        BMINDS_BVEC_FILES, BMINDS_STREAMLINE_FILES,
                                                                        BMINDS_INJECTION_FILES)

    # Join all DWIs with the same region name but different bvals and bvecs using mrtrix
    (CONCATENATED_DWI_LIST, RESIZED_CONCAT_DWI_LIST) = join_dwi_diff_bvals_bvecs(DWI_LIST)

    # Get list of common and uncommon region IDs
    (COMMON_REGION_IDS, NON_COMMON_REGION_IDS) = common_uncommon_regions_list(CONCATENATED_DWI_LIST, STREAMLINE_LIST, INJECTION_LIST)
    
    # FOR COMMON REGIONS - JOIN THE DWI, STREAMLINE AND INJECTION FILES
    for region_ID in COMMON_REGION_IDS:
        # Get the data from the region list function
        (region_list, dwi_found, streamline_found, injection_found) = create_region_list(region_ID, STREAMLINE_LIST, INJECTION_LIST, 
                                                            CONCATENATED_DWI_LIST, BMINDS_ATLAS_FILE, BMINDS_ATLAS_LABEL_FILE, 
                                                            BMINDS_STPT_FILE, BMINDS_MBCA_TRANSFORM_FILE, common_bool=True)
        
        # If we didn't find the dwi, streamline or injection files, skip this region
        if not dwi_found or not streamline_found or not injection_found:
            continue
        
        # Append the subject name, dwi, bval, bvec, streamline and injection to the list
        DATA_LISTS.append(region_list)
    
    # FOR NON-COMMON REGIONS - JOIN THE DWI, STREAMLINE AND INJECTION FILES, where the DWI will be a random one from the list
    for region_ID in NON_COMMON_REGION_IDS:
        # Get the data from the region list function
        (region_list, dwi_found, streamline_found, injection_found) = create_region_list(region_ID, STREAMLINE_LIST, INJECTION_LIST,
                                                            CONCATENATED_DWI_LIST, BMINDS_ATLAS_FILE, BMINDS_ATLAS_LABEL_FILE,
                                                            BMINDS_STPT_FILE, BMINDS_MBCA_TRANSFORM_FILE, common_bool=False)
        
        # If we didn't find the dwi, streamline or injection files, skip this region
        if not dwi_found or not streamline_found or not injection_found:
            continue

        # Append the subject name, dwi, bval, bvec, streamline and injection to the list
        DATA_LISTS.append(region_list)
            
    return DATA_LISTS     

# Create the common and uncommon regions list for the above function
def common_uncommon_regions_list(CONCATENATED_DWI_LIST, STREAMLINE_LIST, INJECTION_LIST):
    # Get the region, or common element ID
    if os.name == "nt":
        DWI_REGION_IDS = [dwi_list[0].split("/")[-3] for dwi_list in CONCATENATED_DWI_LIST]
    else:
        DWI_REGION_IDS = [dwi_list[0].split("/")[-3] for dwi_list in CONCATENATED_DWI_LIST]
    # Get region ID from dwi, streamline and injection files
    STREAMLINE_REGION_IDS = list(set([streamline[0] for streamline in STREAMLINE_LIST]))
    INJECTION_REGION_IDS = list(set([injection[0] for injection in INJECTION_LIST]))

    # Get the common region IDs
    COMMON_REGION_IDS = list(set(DWI_REGION_IDS) & set(STREAMLINE_REGION_IDS) & set(INJECTION_REGION_IDS))

    # Get the region IDs that are not common
    NON_COMMON_REGION_IDS = [region_ID for region_ID in INJECTION_REGION_IDS if region_ID not in COMMON_REGION_IDS]

    return (COMMON_REGION_IDS, NON_COMMON_REGION_IDS)

# Create the list for each region in the above function
def create_region_list(region_ID, STREAMLINE_LIST, INJECTION_LIST, CONCATENATED_DWI_LIST, BMINDS_ATLAS_FILE,
                       BMINDS_ATLAS_LABEL_FILE, BMINDS_STPT_FILE, BMINDS_MBCA_TRANSFORM_FILE, common_bool=True):

    # Booleans for whether we found data or not
    dwi_found = True
    streamline_found = True
    injection_found = True

    # Based on this name, get every streamline and injection that has the same region ID
    streamline_files = [[streamline_file[1], streamline_file[2]] for streamline_file in STREAMLINE_LIST if streamline_file[0] == region_ID]
    injection_files = [[injection_file[1], injection_file[2]] for injection_file in INJECTION_LIST if injection_file[0] == region_ID]
    # Add the atlas, atlas labels and stpt files
    atlas_stpt = [BMINDS_ATLAS_FILE[0], BMINDS_ATLAS_LABEL_FILE[0], BMINDS_STPT_FILE[0]]
    # Add the transform file
    transforms = [BMINDS_MBCA_TRANSFORM_FILE[0]]
    

    # Depending on whether we're processing common or non-common regions, we need to do things differently
    if common_bool:
        # Extract the dwi, bval and bvec files
        dwi_data = [dwi_list for dwi_list in CONCATENATED_DWI_LIST if dwi_list[0].split("/")[-3] == region_ID]
    else:
        # Get a random dwi file
        random_idx = random.randint(0, len(CONCATENATED_DWI_LIST)-1)
        dwi_data = CONCATENATED_DWI_LIST[random_idx]


    # Check that dwi, streamline and injection files are not empty
    if not dwi_data:
        print("No dwi files found for {}".format(region_ID))
        dwi_found = False
    if not streamline_files:
        print("No streamline files found for {}".format(region_ID))
        streamline_found = False
    if not injection_files:
        print("No injection files found for {}".format(region_ID))
        injection_found = False
    
    # Append the subject name, dwi, bval, bvec, streamline and injection to the list
    return [[region_ID, dwi_data, streamline_files, injection_files, atlas_stpt, transforms], dwi_found, streamline_found, injection_found]

# Function to get streamlines from ALL_DATA_LIST
def get_tracer_streamlines_from_all_data_list(ALL_DATA_LIST, streamline_type_list):
    # Define the items to get
    items_to_get = {}
    # Define the allowed types
    allowed_types = ["tracer_tracts_sharp", "dwi_tracts", "tracer_tracts"]
    # Define the streamline index in the ALL_DATA_LIST
    streamline_idx = 2
    
    # Define the items to get based on the type of streamline
    for streamline_type in streamline_type_list:
        # Check that the streamline type is allowed - if yes then append it to dictionary
        if streamline_type in allowed_types:
            items_to_get[streamline_type] = [extract_from_input_list(region[streamline_idx], 
                                            [streamline_type], "streamline")[streamline_type][0] for region in ALL_DATA_LIST]
        else:
            print("Streamline type {} not allowed. Skipping.".format(streamline_type))
            continue

    return items_to_get

# Function to get injections from ALL_DATA_LIST
def get_tracer_injections_from_all_data_list(ALL_DATA_LIST, injection_type_list):
    # Define the items to get
    items_to_get = {}
    # Define the allowed types
    allowed_types = ["tracer_signal_normalized", "tracer_positive_voxels", "cell_density",
                     "streamline_density", "tracer_signal"]
    # Define the injection index in the ALL_DATA_LIST
    injection_idx = 3
    
    # Define the items to get based on the type of injection
    for injection_type in injection_type_list:
        # Check that the injection type is allowed - if yes then append it to dictionary
        if injection_type in allowed_types:
            items_to_get[injection_type] = [extract_from_input_list(region[injection_idx], 
                                            [injection_type], "injection")[injection_type][0] for region in ALL_DATA_LIST]
        else:
            print("Injection type {} not allowed. Skipping.".format(injection_type))
            continue

    return items_to_get

# Function to actually perform the atlas registration and streamline combination commands
def perform_all_general_mrtrix_functions(ALL_DATA_LIST, BMINDS_MBCA_TRANSFORM_FILE, BMINDS_ATLAS_FILE, 
                                            BMINDS_STPT_FILE, BMINDS_ATLAS_LABEL_FILE):
    # Grab the streamlines as a list
    STREAMLINE_TYPES_TO_GRAB = ["tracer_tracts"]
    ALL_STREAMLINES_LIST = get_tracer_streamlines_from_all_data_list(ALL_DATA_LIST, STREAMLINE_TYPES_TO_GRAB)
    # Grab the injections as a list
    INJECTION_TYPES_TO_GRAB = ["cell_density"]
    ALL_INJECTIONS_LIST = get_tracer_injections_from_all_data_list(ALL_DATA_LIST, INJECTION_TYPES_TO_GRAB)
    # Get the transform file
    TRANSFORM_FILE = BMINDS_MBCA_TRANSFORM_FILE[0]
    # Get the atlas and stpt files
    ATLAS_STPT = [BMINDS_ATLAS_FILE[0], BMINDS_ATLAS_LABEL_FILE[0], BMINDS_STPT_FILE[0]]

    # Define the arguments to the function
    ATLAS_STREAMLINE_ARGS = [ALL_STREAMLINES_LIST["tracer_tracts"], ALL_INJECTIONS_LIST["cell_density"], TRANSFORM_FILE, ATLAS_STPT]
    # Get the commands
    MRTRIX_GENERAL_CMDS = mrtrix_all_general_functions(ATLAS_STREAMLINE_ARGS)

    # Run the commands
    for (cmd, cmd_name) in MRTRIX_GENERAL_CMDS:
        print("Started command: {}".format(cmd_name))
        subprocess.run(cmd, shell=True, check=True)

# Function to read the stats file and create a list that has every line
def read_stats_file(STATS_FILES):
    # Holds the data
    STATS_DATA = []
    # Open the file
    for stats_file in STATS_FILES:
        # Load the data with numpy
        stats_data = np.loadtxt(stats_file, delimiter=",")
        # Append the data to the list
        STATS_DATA.append(stats_data)

    return np.array(STATS_DATA)

# Function to get the chosen path for the tracts file
def get_chosen_tracts_stats_folder(REGION_ID, TYPE="includes_both", STATS=False):
    (REGION_MRTRIX_FOLDER, INJECTION_MIF_FOLDER, INJECTION_ROI_TRACTS_FOLDER, INJECTION_ROI_TRACTS_STATS_FOLDER, 
     INJECTION_CONNECTOME_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER, INJECTION_ROI_TRACTS_INCLUDES_ROI_ONLY_FOLDER, 
     INJECTION_ROI_TRACTS_INCLUDES_ENDS_ONLY_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_BOTH_FOLDER, 
     INJECTION_ROI_TRACTS_STATS_INCLUDES_ROI_ONLY_FOLDER, INJECTION_ROI_TRACTS_STATS_INCLUDES_ENDS_ONLY_FOLDER,
     INJECTION_CONNECTOME_INCLUDES_BOTH_FOLDER, INJECTION_CONNECTOME_INCLUDES_ROI_ONLY_FOLDER, 
     INJECTION_CONNECTOME_INCLUDES_ENDS_ONLY_FOLDER) = region_mrtrix_folder_paths(REGION_ID)
    
    # If finding folder for tracts (NOT STATS)
    if not STATS:
        if TYPE == "includes_both":
            CHOSEN_TRACTS_FOLDER = INJECTION_ROI_TRACTS_INCLUDES_BOTH_FOLDER
        elif TYPE == "includes_roi":
            CHOSEN_TRACTS_FOLDER = INJECTION_ROI_TRACTS_INCLUDES_ROI_ONLY_FOLDER
        elif TYPE == "includes_ends_only":
            CHOSEN_TRACTS_FOLDER = INJECTION_ROI_TRACTS_INCLUDES_ENDS_ONLY_FOLDER
        else:
            print("Type {} not allowed. Exiting.".format(TYPE))
            sys.exit()
    # If finding folder for stats
    else:
        if TYPE == "includes_both":
            CHOSEN_TRACTS_FOLDER = INJECTION_ROI_TRACTS_STATS_INCLUDES_BOTH_FOLDER
        elif TYPE == "includes_roi":
            CHOSEN_TRACTS_FOLDER = INJECTION_ROI_TRACTS_STATS_INCLUDES_ROI_ONLY_FOLDER
        elif TYPE == "includes_ends_only":
            CHOSEN_TRACTS_FOLDER = INJECTION_ROI_TRACTS_STATS_INCLUDES_ENDS_ONLY_FOLDER
        else:
            print("Type {} not allowed. Exiting.".format(TYPE))
            sys.exit()

    return CHOSEN_TRACTS_FOLDER

# Function to find the number of streamlines command
def get_tckedit_command(ATLAS_STPT, REGION_ID, ROIS_TO_DO, STREAMLINE_FILE, TYPE="includes_both"):
    # Get the paths to the files
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)
    (COMBINED_TRACTS_PATH) = get_combined_tracts_path()
    (ATLAS_REG_PATH, ATLAS_REG_MIF_PATH) = get_mrtrix_atlas_reg_paths_ants()
    (INJECTION_MIF_PATH) = get_injection_mif_path(REGION_ID)

    # This will store the commands
    TCKEDIT_COMMANDS = []

    # For every ROI, find the number of streamlines between the injection site and the ROI
    for idx, roi_mif_path in enumerate(INDIVIDUAL_ROIS_MIF_PATHS):

        # Get the ROI name or ID
        roi_name = roi_mif_path.split("/")[-1]
        # Check if we need to do this ROI
        if roi_name not in ROIS_TO_DO:
            continue

        # Get the injection ROI tracts path
        (CHOSEN_TRACTS_PATH) = get_injection_roi_tracts_path(REGION_ID, roi_name, TYPE)

        # Find the number of streamlines between the injection site and the ROI
        FIND_STREAMLINES_CMD = define_tckedit_command(COMBINED_TRACTS_PATH, STREAMLINE_FILE, INJECTION_MIF_PATH, ATLAS_REG_MIF_PATH,
                                                        CHOSEN_TRACTS_PATH, TYPE)

        # Add the command to the list
        TCKEDIT_COMMANDS.append(FIND_STREAMLINES_CMD)

    # Return the commands
    return TCKEDIT_COMMANDS

# Function to define the tckedit command depending on the type
def define_tckedit_command(COMBINED_TRACTS_PATH, STREAMLINE_FILE, INJECTION_MIF_PATH, ATLAS_REG_MIF_PATH, 
                           CHOSEN_TRACTS_PATH, TYPE="includes_both"):
    # Get the paths

    # Define the command depending on the type
    if TYPE == "includes_both":
        FIND_STREAMLINES_CMD = "tckedit {all_tracts}.tck -include {inj_site}.mif -include {atlas_roi}.mif {output}.tck -force".format(
            all_tracts=COMBINED_TRACTS_PATH, inj_site=INJECTION_MIF_PATH, atlas_roi=ATLAS_REG_MIF_PATH, 
            output=CHOSEN_TRACTS_PATH)
    elif TYPE == "includes_roi":
        FIND_STREAMLINES_CMD = "tckedit {individual_tract} -include {atlas_roi}.mif {output}.tck -force".format(
            individual_tract=STREAMLINE_FILE, inj_site=INJECTION_MIF_PATH, atlas_roi=ATLAS_REG_MIF_PATH, 
            output=CHOSEN_TRACTS_PATH)
    elif TYPE == "includes_ends_only":
        FIND_STREAMLINES_CMD = "tckedit {individual_tract} -include {inj_site}.mif {output}.tck -ends_only -force".format(
            individual_tract=STREAMLINE_FILE, inj_site=INJECTION_MIF_PATH, atlas_roi=ATLAS_REG_MIF_PATH, 
            output=CHOSEN_TRACTS_PATH)
    else:
        print("Type {} not allowed. Exiting.".format(TYPE))
        sys.exit()

    return FIND_STREAMLINES_CMD
    
# Function to find the stats command
def get_tckstats_command(ATLAS_STPT, REGION_ID, ROIS_TO_DO, TYPE="includes_both"):

    # Get the paths
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)

    # This will hold all the commands
    TCKSTATS_COMMANDS = []

    # For every ROI, find the stats of the number of streamlines between the injection site and the ROI
    for idx, roi_mif_path in enumerate(INDIVIDUAL_ROIS_MIF_PATHS):

        # Get the ROI name or ID
        roi_name = roi_mif_path.split("/")[-1]
        # Check if we need to do this ROI
        if roi_name not in ROIS_TO_DO:
            continue

        # Get the injection ROI tracts and stats path
        (CHOSEN_TRACTS_PATH) = get_injection_roi_tracts_path(REGION_ID, roi_name, TYPE)
        (INJECTION_ROI_LENGTHS_PATH, INJECTION_ROI_COUNT_PATH, INJECTION_ROI_MEAN_PATH, 
         INJECTION_ROI_MEDIAN_PATH, INJECTION_ROI_STD_PATH, INJECTION_ROI_MIN_PATH, 
         INJECTION_ROI_MAX_PATH) = get_injection_roi_tracts_stats_path(REGION_ID, roi_name, TYPE="includes_both")

        # Find the stats of the number of streamlines between the injection site and the ROI. Note that it PRINTS out
        # everything, but we can grab the count by counting the number of lines in the file
        # Can also get the mean, median, min, max, std, etc. by doing other modifications on the text file
        FIND_STATS_CMD = "tckstats {input}.tck -dump {output} -force".format(input=CHOSEN_TRACTS_PATH, 
                                                                        output=INJECTION_ROI_LENGTHS_PATH)
        
        # Add the command to the list
        TCKSTATS_COMMANDS.append(FIND_STATS_CMD)

    # Return the commands
    return TCKSTATS_COMMANDS

# Function to actually call the stats and save them to files
def find_and_save_stats_results(ATLAS_STPT, REGION_ID, TYPE="includes_both"):

    # Get the paths
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)

    # For every ROI, find the stats of the number of streamlines between the injection site and the ROI
    for idx, roi_mif_path in enumerate(INDIVIDUAL_ROIS_MIF_PATHS):
        
        # Get the ROI name or ID
        roi_name = roi_mif_path.split("/")[-1]
        # Get the injection ROI stats path
        (INJECTION_ROI_LENGTHS_PATH, INJECTION_ROI_COUNT_PATH, INJECTION_ROI_MEAN_PATH, 
         INJECTION_ROI_MEDIAN_PATH, INJECTION_ROI_STD_PATH, INJECTION_ROI_MIN_PATH, 
         INJECTION_ROI_MAX_PATH) = get_injection_roi_tracts_stats_path(REGION_ID, roi_name, TYPE)
            
        # Get all the stats about the file
        count_data = get_stats_from_file(INJECTION_ROI_LENGTHS_PATH, TYPE="count")
        mean_data = get_stats_from_file(INJECTION_ROI_LENGTHS_PATH, TYPE="mean")
        median_data = get_stats_from_file(INJECTION_ROI_LENGTHS_PATH, TYPE="median")
        min_data = get_stats_from_file(INJECTION_ROI_LENGTHS_PATH, TYPE="min")
        max_data = get_stats_from_file(INJECTION_ROI_LENGTHS_PATH, TYPE="max")
        std_data = get_stats_from_file(INJECTION_ROI_LENGTHS_PATH, TYPE="std")

        # Save the count, mean, median, min, max, std, etc. to the text file using numpy
        np.savetxt(INJECTION_ROI_COUNT_PATH, count_data, delimiter=",")
        np.savetxt(INJECTION_ROI_MEAN_PATH, mean_data, delimiter=",")
        np.savetxt(INJECTION_ROI_MEDIAN_PATH, median_data, delimiter=",")
        np.savetxt(INJECTION_ROI_MIN_PATH, min_data, delimiter=",")
        np.savetxt(INJECTION_ROI_MAX_PATH, max_data, delimiter=",")
        np.savetxt(INJECTION_ROI_STD_PATH, std_data, delimiter=",")
            

# Function to get the stats (median, mean, etc) from a file
def get_stats_from_file(STATS_FILE, TYPE="count"):

    # Load the file with numpy
    lengths_data = np.loadtxt(STATS_FILE)

    if len(lengths_data) == 0:
        print("Lengths data {} is empty. Exiting.".format(STATS_FILE))
        # Save to file that it's empty
        EMPTY_FILE = STATS_FILE.replace(".txt", "_empty.txt")
        np.savetxt(EMPTY_FILE, lengths_data, delimiter=",")
        sys.exit()

    # Get the stats from the file, depending on the type
    if TYPE == "count":
        count_data = np.array(len(lengths_data))
        return count_data
    elif TYPE == "lengths":
        return lengths_data
    elif TYPE == "mean":
        mean_data = np.array(float(sum(lengths_data))/len(lengths_data) if len(lengths_data) > 0 else float('nan'))
        return mean_data
    elif TYPE == "median":
        median_data = np.array(float(sorted(lengths_data)[len(lengths_data)//2]) if len(lengths_data) > 0 else float('nan'))
        return median_data
    elif TYPE == "min":
        min_data = np.array(float(min(lengths_data)) if len(lengths_data) > 0 else float('nan'))
        return min_data
    elif TYPE == "max":
        max_data = np.array(float(max(lengths_data)) if len(lengths_data) > 0 else float('nan'))
        return max_data
    elif TYPE == "std":
        std_data = np.array(np.std(lengths_data) if len(lengths_data) > 0 else float('nan'))
        return std_data
    else:
        print("Type {} not allowed. Exiting.".format(TYPE))
        sys.exit()


# TO MAKE ATLAS
# LOAD EACH NIFTI COPY TO MAKE "ATLAS" THEN FOR LOOP AND WHEREVER THE NIFTI IS NON-ZERO SET THAT REGION IN "ATLAS" TO I
#MRICRO MAKE ACTUAL ATLAS w injections if ya 