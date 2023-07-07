import os

import sys
sys.path.append("..")
from py_helpers.general_helpers import *
from py_helpers.shared_helpers import *
from .inj_commands import *
from numpy import random

# Create a list that associates each subject with its T1 and DWI files
def create_data_list(BMINDS_UNZIPPED_DWI_FILES, BMINDS_BVAL_FILES, BMINDS_BVEC_FILES, BMINDS_STREAMLINE_FILES, 
                     BMINDS_INJECTION_FILES, BMINDS_ATLAS_FILE, BMINDS_STPT_FILE, BMINDS_MBCA_TRANSFORM_FILE):
    
    # This will hold all of the data lists
    DATA_LISTS = []
    
    # Get the initial lists
    (DWI_LIST, STREAMLINE_LIST, INJECTION_LIST) = create_initial_lists(BMINDS_UNZIPPED_DWI_FILES, BMINDS_BVAL_FILES,
                                                                        BMINDS_BVEC_FILES, BMINDS_STREAMLINE_FILES,
                                                                        BMINDS_INJECTION_FILES)

    # Join all DWIs with the same region name but different bvals and bvecs using mrtrix
    CONCATENATED_DWI_LIST = join_dwi_diff_bvals_bvecs(DWI_LIST)

    # Get list of common and uncommon region IDs
    (COMMON_REGION_IDS, NON_COMMON_REGION_IDS) = common_uncommon_regions_list(CONCATENATED_DWI_LIST, STREAMLINE_LIST, INJECTION_LIST)

    print('Length of common region IDs: ', len(COMMON_REGION_IDS))
    print('Length of non-common region IDs: ', len(NON_COMMON_REGION_IDS))
    
    # FOR COMMON REGIONS - JOIN THE DWI, STREAMLINE AND INJECTION FILES
    for region_ID in COMMON_REGION_IDS:
        # Get the data from the region list function
        (region_list, dwi_found, streamline_found, injection_found) = create_region_list(region_ID, STREAMLINE_LIST, INJECTION_LIST, 
                                                            CONCATENATED_DWI_LIST, BMINDS_ATLAS_FILE, BMINDS_STPT_FILE, 
                                                            BMINDS_MBCA_TRANSFORM_FILE, common_bool=True)
        
        # If we didn't find the dwi, streamline or injection files, skip this region
        if not dwi_found or not streamline_found or not injection_found:
            continue
        
        # Append the subject name, dwi, bval, bvec, streamline and injection to the list
        DATA_LISTS.append(region_list)
    
    # FOR NON-COMMON REGIONS - JOIN THE DWI, STREAMLINE AND INJECTION FILES, where the DWI will be a random one from the list
    for region_ID in NON_COMMON_REGION_IDS:
        # Get the data from the region list function
        (region_list, dwi_found, streamline_found, injection_found) = create_region_list(region_ID, STREAMLINE_LIST, INJECTION_LIST,
                                                            CONCATENATED_DWI_LIST, BMINDS_ATLAS_FILE, BMINDS_STPT_FILE, 
                                                            BMINDS_MBCA_TRANSFORM_FILE, common_bool=False)
        
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
                       BMINDS_STPT_FILE, BMINDS_MBCA_TRANSFORM_FILE, common_bool=True):

    # Booleans for whether we found data or not
    dwi_found = True
    streamline_found = True
    injection_found = True

    # Based on this name, get every streamline and injection that has the same region ID
    streamline_files = [[streamline_file[1], streamline_file[2]] for streamline_file in STREAMLINE_LIST if streamline_file[0] == region_ID]
    injection_files = [[injection_file[1], injection_file[2]] for injection_file in INJECTION_LIST if injection_file[0] == region_ID]
    # Add the atlas and stpt files
    atlas_stpt = [BMINDS_ATLAS_FILE[0], BMINDS_STPT_FILE[0]]
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
    
    # Define the items to get based on the type of streamline
    for streamline_type in streamline_type_list:
        # Check that the streamline type is allowed - if yes then append it to dictionary
        if streamline_type in allowed_types:
            items_to_get[streamline_type] = [extract_from_input_list(region[2], [streamline_type], "streamline")[streamline_type][0] for region in ALL_DATA_LIST]
        else:
            print("Streamline type {} not allowed. Skipping.".format(streamline_type))
            continue

    return items_to_get

# Function to actually perform the atlas registration and streamline combination commands
def perform_atlas_registration_and_streamline_combo(ALL_DATA_LIST, BMINDS_MBCA_TRANSFORM_FILE, BMINDS_ATLAS_FILE, 
                                                    BMINDS_STPT_FILE):
    STREAMLINE_TYPES_TO_GRAB = ["tracer_tracts"]
    ALL_STREAMLINES_LIST = get_tracer_streamlines_from_all_data_list(ALL_DATA_LIST, STREAMLINE_TYPES_TO_GRAB)
    TRANSFORM_FILE = BMINDS_MBCA_TRANSFORM_FILE[0]
    ATLAS_STPT = [BMINDS_ATLAS_FILE[0], BMINDS_STPT_FILE[0]]
    ATLAS_STREAMLINE_ARGS = [ALL_STREAMLINES_LIST["tracer_tracts"], TRANSFORM_FILE, ATLAS_STPT]
    # Get the commands
    STREAMLINE_ATLAS_CMDS = mrtrix_atlas_registration_and_streamline_combination(ATLAS_STREAMLINE_ARGS)
    # Run the commands
    for (cmd, cmd_name) in STREAMLINE_ATLAS_CMDS:
        print("Started {} - {}".format(cmd_name, "common"))
        subprocess.run(cmd, shell=True, check=True)
    
    