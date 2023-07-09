# Focus right now is to
# 1. Make the injections cell density files into masks
# 2. Combine all tract-tracing streamlines into one file
# 3. See which streamlines in which ROI interact with the injection site
# 4. Maybe we can make a mask of injection site per ROI, and then see which streamlines interact with that mask
# And do that for every injection site and every ROI

# OPTION 1: Create one big mask of BOTH all the injection sites and all the ROIs (though there will be a lot of overlap),
# and then see which streamlines interact with that mask and then get a subset of injection -> ROI only
# OPTION 2: Create a mask of each injection site and each ROI, and then see which streamlines interact with that mask

# The way the authors do it is by first making a normal connectome of IM ASSUMING all ROIs, then doing this
# def mapme(my_path_source,CM): #function for mapping to 20 x 104 tracer-based matrix (Code and tracer-based connectome from Skibbe H.)
    # mapping = sio.loadmat(my_path_source+'atlas/'+'mat_mapping.mat')
    # # the injection site regions we have in meso
    # macro_srcs = np.squeeze(mapping['macro_srcs'])
    # macro_srcs -= 1 # matlab2pyhton indexing
    # #the mapping from high res atlas to low res atlas
    # all_src_maps = np.squeeze(mapping['all_src_maps'])
    # # number of unique targets - background
    # valid_targets = np.unique(all_src_maps).shape[0]-1
    # CM_macro = np.zeros((CM.shape[0],valid_targets))
    # #run over all targets
    # for b in range(1,valid_targets+1):    
    #     CM_macro[:,b-1] = np.sum(CM[:,all_src_maps == b],axis=1)    
    # CMma = CM_macro[macro_srcs,:]
    # CMma_norm = mb.repmat(np.sum(CMma,axis=1,keepdims=True),1,CMma.shape[1])
    # CMma  = np.divide(CMma , CMma_norm)
    # CMma_all = CM[macro_srcs,:]
    # CMma_norm = mb.repmat(np.sum(CMma_all,axis=1,keepdims=True),1,CMma_all.shape[1])
    # CMma_all  = np.divide(CMma_all , CMma_norm)
    # return CMma, CMma_all

import os
import sys
sys.path.append("..")
from .inj_paths import *
from .inj_checkpoints import *
from py_helpers.shared_helpers import *

# Function to use the transforms h5 file given, with ants
def use_transforms_h5_file(ARGS):

    # Extract arguments needed to define paths
    TRANSFORMS_H5 = ARGS[0]
    ATLAS_STPT = ARGS[1]

    print("TRANSFORM_H5: {}".format(TRANSFORMS_H5))

    # Define what's needed for the commands
    NEEDED_FILES_TRANSFORM = ["mbca_transform"]
    TRANSFORM_NEEDED_PATH = extract_from_input_list(TRANSFORMS_H5, NEEDED_FILES_TRANSFORM, "transforms")
    NEEDED_FILES_ATLAS = ["atlas", "stpt"]
    ATLAS_STPT_NEEDED_PATH = extract_from_input_list(ATLAS_STPT, NEEDED_FILES_ATLAS, "atlas_stpt")

    # Get the rest of the paths for the commands
    (ATLAS_REG_PATH, ATLAS_REG_MIF_PATH) = get_mrtrix_atlas_reg_paths_ants()

    # Register the atlas to the STPT DWI space using the transformation
    REGISTER_ATLAS_DWI_CMD = "antsApplyTransforms -d 3 -i {want_to_register} -r {register_to} -o {output}.nii.gz -t {transform}".format(
        want_to_register=ATLAS_STPT_NEEDED_PATH["atlas"], register_to=ATLAS_STPT_NEEDED_PATH["stpt"], output=ATLAS_REG_PATH,
        transform=TRANSFORM_NEEDED_PATH["mbca_transform"])
    # Convert the atlas to mif
    CONVERT_ATLAS_TO_MIF_CMD = "mrconvert {input}.nii.gz {output}.mif".format(input=ATLAS_REG_PATH, output=ATLAS_REG_MIF_PATH)

    # Return the commands
    return (REGISTER_ATLAS_DWI_CMD, CONVERT_ATLAS_TO_MIF_CMD)

# Function to combine all the streamline files into one file
def combine_all_streamline_files(ARGS):

    # Extract arguments needed to define paths
    STREAMLINE_FILES = ARGS[0]
    
    # Create a string of all the streamline files
    STREAMLINE_FILES_STRING = " ".join(STREAMLINE_FILES)

    # Get the combined tracts path
    COMBINED_TRACTS_PATH = get_combined_tracts_path()

    # Combine all the streamline files into one file
    COMBINE_STREAMLINE_CMD = "tckedit {input} {output}.tck".format(input=STREAMLINE_FILES_STRING, output=COMBINED_TRACTS_PATH)
    
    # Return the command
    return (COMBINE_STREAMLINE_CMD)

# Function to combine all the injection files into one file
def combine_all_injection_files(ARGS):

    # Extract arguments needed to define paths
    INJECTION_FILES = ARGS[0]

    # Create a string of all the injection files
    INJECTION_FILES_STRING = " ".join(INJECTION_FILES)

    # Get the combined injections path
    COMBINED_INJECTIONS_PATH = get_combined_injections_path()

    # Combine all the injection files into one file
    COMBINE_INJECTION_CMD = "mrcat {input} {output}.nii.gz".format(input=INJECTION_FILES_STRING, output=COMBINED_INJECTIONS_PATH)

    # Return the command
    return (COMBINE_INJECTION_CMD)

# Function to do ALL the injection combination
def mrtrix_injection_combination(ARGS):

    # Extract arguments needed to define paths
    INJECTION_FILES = ARGS[0]

    # Define the injection combination command
    INJECTION_COMBO_ARGS = [INJECTION_FILES]
    (COMBINE_INJECTION_CMD) = combine_all_injection_files(INJECTION_COMBO_ARGS)

    # Check if we need to do the above commands
    (MRTRIX_INJECTION_COMBINATION) = check_missing_injection_all()

    # Create MRTRIX commands, depending on what we need to do
    MRTRIX_COMMANDS = []
    if MRTRIX_INJECTION_COMBINATION:
        MRTRIX_COMMANDS.extend([
            (COMBINE_INJECTION_CMD, "Combining all injection files")
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

# Function to do the atlas registration and streamline combination
def mrtrix_all_general_functions(ARGS):

    # Extract arguments needed to define paths
    STREAMLINE_FILES = ARGS[0]
    INJECTION_FILES = ARGS[1]
    TRANSFORMS_H5 = ARGS[2]
    ATLAS_STPT = ARGS[3]

    # Get the main paths (only one will be used)
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, 
        INDIVIDUAL_ROIS_MIF_FOLDER_NAME, INJECTION_ROI_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Define the atlas registration commands
    ATLAS_REG_ARGS = [TRANSFORMS_H5, ATLAS_STPT]
    (REGISTER_ATLAS_DWI_CMD, CONVERT_ATLAS_TO_MIF_CMD) = use_transforms_h5_file(ATLAS_REG_ARGS)

    # Define the streamline combination command
    STREAMLINE_COMBO_ARGS = [STREAMLINE_FILES]
    (COMBINE_STREAMLINE_CMD) = combine_all_streamline_files(STREAMLINE_COMBO_ARGS)

    # Extract the ROIs of each atlas
    ATLAS_ROI_ARGS = [ATLAS_STPT, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME]
    (EXTRACTION_COMMANDS) = extract_each_roi_from_atlas(ATLAS_ROI_ARGS)

    # Convert each ROI to mif
    CONVERT_ROI_TO_MIF_CMD = convert_each_roi_to_mif(ATLAS_STPT)

    # Check if we need to do the above commands
    CHECK_MISSING_GENERAL_ARGS = [ATLAS_STPT]
    (MRTRIX_ATLAS_REGISTRATION, MRTRIX_STREAMLINE_COMBINATION, 
        MRTRIX_ATLAS_ROIS, MRTRIX_ATLAS_MIF_CONVERSION) = check_missing_general_files(CHECK_MISSING_GENERAL_ARGS)

    # Create MRTRIX commands, depending on what we need to do
    MRTRIX_COMMANDS = []
    if MRTRIX_ATLAS_REGISTRATION:
        MRTRIX_COMMANDS.extend([
            (REGISTER_ATLAS_DWI_CMD, "Registering atlas to STPT DWI using ANTs"),
            (CONVERT_ATLAS_TO_MIF_CMD, "Converting atlas to mif")
        ])
    if MRTRIX_STREAMLINE_COMBINATION:
        MRTRIX_COMMANDS.extend([
            (COMBINE_STREAMLINE_CMD, "Combining all streamline files")
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

    # Return the commands
    return (MRTRIX_COMMANDS)

# Function to do all the mrtrix commands for each individual file rather than all
def mrtrix_all_region_functions(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    DWI_FILE = ARGS[1]
    STREAMLINE_FILE = ARGS[2]
    INJECTION_FILE = ARGS[3]
    ATLAS_STPT = ARGS[4]

    # Define the injection mif commands
    INJECTION_MIF_ARGS = [REGION_ID, INJECTION_FILE]
    (CREATE_INJECTION_MIF_CMD) = create_mifs_of_each_injection_site(INJECTION_MIF_ARGS)

    # Define the injection and ROI combination commands
    INJECTION_ROI_ARGS = [REGION_ID, ATLAS_STPT]
    (COMBINATION_COMMANDS) = combine_each_injection_site_mif_with_each_roi_mif(INJECTION_ROI_ARGS)

    # Check if we need to do the above commands
    CHECKPOINT_ARGS = [REGION_ID, ATLAS_STPT]
    (INJECTION_MIFS, INJECTION_ROI_COMBINATION) = check_missing_region_files(CHECKPOINT_ARGS)

    # Create MRTRIX commands, depending on what we need to do
    MRTRIX_COMMANDS = []
    if INJECTION_MIFS:
        MRTRIX_COMMANDS.extend([
            (CREATE_INJECTION_MIF_CMD, "Creating injection mifs")
        ])
    if INJECTION_ROI_COMBINATION:
        for idx, combine in COMBINATION_COMMANDS:
            MRTRIX_COMMANDS.extend([
                (combine, "Combining injection mif with ROI mif {}".format(idx))
            ])

    # Return the commands
    return (MRTRIX_COMMANDS)

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

    # Get the main folders
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, 
        INDIVIDUAL_ROIS_MIF_FOLDER_NAME, INJECTION_ROI_FOLDER_NAME) = main_mrtrix_folder_paths()
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
def convert_each_roi_to_mif(ATLAS_STPT):

    # This will hold all the conversion commands
    CONVERSION_COMMANDS = []

    # Get the main path for the atlas ROIs
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
        COMBINED_INJECTIONS_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, 
        INDIVIDUAL_ROIS_MIF_FOLDER_NAME, INJECTION_ROI_FOLDER_NAME) = main_mrtrix_folder_paths()
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

# Function to combine each injection site mif with each ROI mif
def combine_each_injection_site_mif_with_each_roi_mif(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]

    # Get all the injection mif paths
    (INJECTION_MIF_PATH) = get_injection_mif_path(REGION_ID)

    # Get all the atlas ROI mif paths
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)

    # This will hold all the combination commands
    COMBINATION_COMMANDS = []

    # Combine the injection and ROI mifs
    for roi_mif_filepath in INDIVIDUAL_ROIS_MIF_PATHS:
        # Get the output filename
        roi_filename = roi_mif_filepath.split("/")[-1]
        # Get the injection ROI mif path
        INJECTION_ROI_MIF_PATH = get_injection_roi_path(REGION_ID, roi_filename)
        # Combine the injection and ROI mifs
        COMBINE_INJECTION_ROI_MIF_CMD = "mrcat {inj_mif}.mif {roi_mif}.mif {output}.mif".format(
            inj_mif=INJECTION_MIF_PATH, roi_mif=roi_mif_filepath, output=INJECTION_ROI_MIF_PATH)
        
        # Add the command to the list
        COMBINATION_COMMANDS.append(COMBINE_INJECTION_ROI_MIF_CMD)
        
    # Return the command
    return (COMBINATION_COMMANDS)


# TODO:
# 1. Find some way to find streamlines between injection sites and atlas ROIs - atlas only needs to be a mif, not a mask!!
# To do this, we can do
# 1. Create a function to extract all the individual ROIs in the 140 atlas -------------------------------- DONE 
# 2. Create a function to convert all the individual ROIs in the 140 atlas to mifs ------------------------ DONE 
# 3. Create a function to convert all the injection sites to mifs ----------------------------------------- DONE 
# 4. Create a function to combine each injection site mif with each ROI mif
# 5. Create a function to find streamlines between these injection <-> ROI mif combinations
# 6. Create the connectome of this for each injection <-> ROI mif combination
# 7. Create a function to combine all the connectomes into one big connectome

