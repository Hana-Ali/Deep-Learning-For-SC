import os
import sys
sys.path.append("..")
from .inj_paths import *
from .inj_checkpoints import *
from py_helpers.shared_helpers import *
import nibabel as nib

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
        COMBINED_INJECTIONS_FOLDER_NAME, COMBINED_ATLAS_INJECTIONS_FOLDER_NAME, COMBINED_CONNECTOME_FOLDER_NAME,
        INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME, INDIVIDUAL_ROIS_NIFTI_FOLDER_NAME, INDIVIDUAL_ROIS_MIF_FOLDER_NAME, 
        INJECTION_ROI_FOLDER_NAME, INJECTION_ROI_CONNECTOMES_FOLDER_NAME) = main_mrtrix_folder_paths()

    # Extract the ROIs of each atlas
    ATLAS_ROI_ARGS = [ATLAS_STPT, INDIVIDUAL_ROIS_FROM_ATLAS_FOLDER_NAME]
    (EXTRACTION_COMMANDS) = extract_each_roi_from_atlas(ATLAS_ROI_ARGS)

    # Convert each ROI to mif
    ROI_MIF_ARGS = [ATLAS_STPT]
    CONVERT_ROI_TO_MIF_CMD = convert_each_roi_to_mif(ROI_MIF_ARGS)

    # Define the injection mif commands
    INJECTION_MIF_ARGS = [REGION_ID, INJECTION_FILE]
    (CREATE_INJECTION_MIF_CMD) = create_mifs_of_each_injection_site(INJECTION_MIF_ARGS)

    # Define the injection and ROI combination commands
    INJECTION_ROI_ARGS = [REGION_ID, ATLAS_STPT]
    (COMBINATION_COMMANDS) = combine_each_injection_site_mif_with_each_roi_mif(INJECTION_ROI_ARGS)

    # Define the connectome creation commands
    CONNECTOME_ARGS = [REGION_ID, ATLAS_STPT]
    (CONNECTOME_COMMANDS) = create_connectome_for_each_injection_roi_combination(CONNECTOME_ARGS)

    # Check if we need to do the above commands
    CHECKPOINT_ARGS = [REGION_ID, ATLAS_STPT]
    (INJECTION_MIFS, MRTRIX_ATLAS_ROIS, MRTRIX_ATLAS_MIF_CONVERSION, INJECTION_ROI_COMBINATION, 
     CONNECTOMES) = check_missing_region_files(CHECKPOINT_ARGS)

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
    if INJECTION_ROI_COMBINATION:
        for idx, combine in enumerate(COMBINATION_COMMANDS):
            MRTRIX_COMMANDS.extend([
                (combine, "Combining injection mif with ROI mif {}".format(idx))
            ])
    if CONNECTOMES:
        for idx, connectome in enumerate(CONNECTOME_COMMANDS):
            MRTRIX_COMMANDS.extend([
                (connectome, "Creating connectome for injection <-> ROI combination {}".format(idx))
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

# Function to create connectome for each injection <-> ROI combination
def create_connectome_for_each_injection_roi_combination(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]

    # Get all the atlas ROI mif paths
    (INDIVIDUAL_ROIS_MIF_PATHS) = get_individual_rois_mif_path(ATLAS_STPT)
    
    # Get the concatenated streamlines path
    (COMBINED_TRACTS_PATH) = get_combined_tracts_path()

    # This will hold all the connectome commands
    CONNECTOME_COMMANDS = []

    # Create the connectome for each injection <-> ROI combination
    for roi_mif_filepath in INDIVIDUAL_ROIS_MIF_PATHS:
        # Get the output filename
        roi_filename = roi_mif_filepath.split("/")[-1]
        # Get the injection ROI mif path
        INJECTION_ROI_MIF_PATH = get_injection_roi_path(REGION_ID, roi_filename)
        # Get the connectome path
        INJECTION_ROI_CONNECTOME_PATH = get_injection_roi_connectome_path(REGION_ID, roi_filename)
        # Create the connectome
        CREATE_CONNECTOME_CMD = "tck2connectome {input}.tck {nodes}.mif {output}.csv -symmetric -zero_diagonal".format(
            input=COMBINED_TRACTS_PATH, nodes=INJECTION_ROI_MIF_PATH, output=INJECTION_ROI_CONNECTOME_PATH)
        # Add the command to the list
        CONNECTOME_COMMANDS.append(CREATE_CONNECTOME_CMD)

    # Return the commands
    return (CONNECTOME_COMMANDS)    
