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
import numpy as np

from .inj_paths import *
from .inj_checkpoints import *

from py_helpers import *

# Function to use the transforms h5 file given, with ants
def use_transforms_h5_file(ARGS):

    # Extract arguments needed to define paths
    TRANSFORMS_H5 = ARGS[0]
    ATLAS_STPT = ARGS[1]

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


# Function to do the atlas registration and streamline combination
def mrtrix_all_general_functions(ARGS):

    # Extract arguments needed to define paths
    STREAMLINE_FILES = ARGS[0]
    INJECTION_FILES = ARGS[1]
    TRANSFORMS_H5 = ARGS[2]
    ATLAS_STPT = ARGS[3]

    # Define the atlas registration commands
    ATLAS_REG_ARGS = [TRANSFORMS_H5, ATLAS_STPT]
    (REGISTER_ATLAS_DWI_CMD, CONVERT_ATLAS_TO_MIF_CMD) = use_transforms_h5_file(ATLAS_REG_ARGS)

    # Define the streamline combination command
    STREAMLINE_COMBO_ARGS = [STREAMLINE_FILES["tracer_tracts"]]
    (COMBINE_STREAMLINE_CMD) = combine_all_streamline_files(STREAMLINE_COMBO_ARGS)

    # Define the streamline -> TDI command
    STREAMLINE_TDI_ARGS = [STREAMLINE_FILES, INJECTION_FILES["streamline_density"]]
    (STREAMLINE_TDI_CMD) = create_tdi_from_streamlines(STREAMLINE_TDI_ARGS)

    # Check if we need to do the above commands
    CHECK_MISSING_GENERAL_ARGS = [ATLAS_STPT]
    (MRTRIX_ATLAS_REGISTRATION, MRTRIX_STREAMLINE_COMBINATION) = check_missing_general_files(CHECK_MISSING_GENERAL_ARGS)

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

    # Return the commands
    return (MRTRIX_COMMANDS)

# Function to take all the matrices in each region file and combine them into one big matrix
def combine_all_region_stats():

    # Get the main paths
    (GENERAL_MRTRIX_FOLDER, SPECIFIC_MRTRIX_FOLDER, ATLAS_REG_FOLDER_NAME, COMBINED_TRACTS_FOLDER_NAME, 
    COMBINED_CONNECTOME_FOLDER_NAME, DENSITY_MAPS_FOLDER_NAME, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, 
    INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME) = main_mrtrix_folder_paths()
    
    # Get all the text names in SPECIFIC_MRTRIX_FOLDER and filter for ones with vector in the name
    REGION_FOLDER_NAMES = glob_files(SPECIFIC_MRTRIX_FOLDER, "txt")
    REGION_FOLDER_NAMES = [file for file in REGION_FOLDER_NAMES if "vector" in file]
    # Ensure it isn't empty and that its length is 52 (we have 52 injections)
    check_globbed_files(REGION_FOLDER_NAMES, "region files")
    if len(REGION_FOLDER_NAMES) != 52:
        print("Not all region files were found. Please check that all 52 region files are in the folder")
        sys.exit('Exiting program')

    # Get the combined connectome path
    (COMBINED_CONNECTOME_PATH) = get_combined_connectome_path()

    # Load the data from all the txt files into a numpy array
    ALL_REGION_DATA = []
    for region_file in REGION_FOLDER_NAMES:
        REGION_DATA = np.loadtxt(region_file)
        ALL_REGION_DATA.append(REGION_DATA)
    
    # Combine all the data into one big matrix
    ALL_REGION_DATA = np.concatenate(ALL_REGION_DATA, axis=1)

    # Save the data into a csv file
    np.savetxt(COMBINED_CONNECTOME_PATH, ALL_REGION_DATA, delimiter=",")

# Function to make the tracer type
def get_tracer_type(streamline_file):
    return streamline_file.split(os.sep)[-1]

# Function to convert all tracer, tracer_sharp, and dwi tracts to TDI
def create_tdi_from_streamlines(ARGS):

    # Get the arguments
    STREAMLINE_FILES = ARGS[0]
    INJECTION_FILES = ARGS[1]

    # Get the different streamline files
    TRACER_STREAMLINE_FILES = STREAMLINE_FILES["tracer_tracts"]
    TRACER_SHARP_STREAMLINE_FILES = STREAMLINE_FILES["tracer_sharp_tracts"]
    DWI_STREAMLINE_FILES = STREAMLINE_FILES["dwi_tracts"]

    # Get the first one as a template
    TEMPLATE = INJECTION_FILES[0]

    print("template is: ", TEMPLATE)

    # This stores the commands as a dictionary
    TDI_COMMANDS = {}

    # For every tracer streamline file, create a TDI
    for tracer_streamline in TRACER_STREAMLINE_FILES:
        # Get the type
        TRACER_TYPE = get_tracer_type(tracer_streamline)
        # Get the TDI path
        (TDI_REGION_FOLDER, TDI_PATH) = get_tdi_path(tracer_streamline)
        # Create the TDI command
        TDI_CMD = "tckmap {input} {output}.nii.gz -template {template}".format(input=tracer_streamline, 
                                                                               output=TDI_PATH, 
                                                                               template=TEMPLATE)
        # Append the command to the dictionary
        TDI_COMMANDS[TRACER_TYPE] = TDI_CMD

    # For every tracer_sharp streamline file, create a TDI
    for tracer_sharp_streamline in TRACER_SHARP_STREAMLINE_FILES:
        # Get the type
        TRACER_TYPE = get_tracer_type(tracer_sharp_streamline)
        # Get the TDI path
        (TDI_REGION_FOLDER, TDI_PATH) = get_tdi_path(tracer_sharp_streamline)
        # Create the TDI command
        TDI_CMD = "tckmap {input} {output}.nii.gz -template {template}".format(input=tracer_sharp_streamline, 
                                                                               output=TDI_PATH, 
                                                                               template=TEMPLATE)
        # Append the command to the dictionary
        TDI_COMMANDS[TRACER_TYPE] = TDI_CMD

    # For every dwi streamline file, create a TDI
    for dwi_streamline in DWI_STREAMLINE_FILES:
        # Get the type
        TRACER_TYPE = get_tracer_type(dwi_streamline)
        # Get the TDI path
        (TDI_REGION_FOLDER, TDI_PATH) = get_tdi_path(dwi_streamline)
        # Create the TDI command
        TDI_CMD = "tckmap {input} {output}.nii.gz -template {template}".format(input=dwi_streamline, 
                                                                               output=TDI_PATH, 
                                                                               template=TEMPLATE)
        # Append the command to the dictionary
        TDI_COMMANDS[TRACER_TYPE] = TDI_CMD

    # Return the commands
    return (TDI_COMMANDS)
    


# Function to symmetrize all the tracer, tracer_sharp, and dwi tracts TDIs











# TODO:
# 1. Find some way to find streamlines between injection sites and atlas ROIs - atlas only needs to be a mif, not a mask!!
# To do this, we can do
# 1. Create a function to extract all the individual ROIs in the 140 atlas -------------------------------- DONE 
# 2. Create a function to convert all the individual ROIs in the 140 atlas to mifs ------------------------ DONE 
# 3. Create a function to convert all the injection sites to mifs ----------------------------------------- DONE 
# 4. Create a function to combine each injection site mif with each ROI mif ------------------------------- DONE 
# 5. Create a function to find streamlines between these injection <-> ROI mif combinations --------------- DONE
# 6. Create the connectome of this for each injection <-> ROI mif combination ----------------------------- DONE 
# 7. Create a function to combine all the connectomes into one big connectome
