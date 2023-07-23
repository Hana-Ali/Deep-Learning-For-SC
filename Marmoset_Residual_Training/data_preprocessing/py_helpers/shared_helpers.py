import os
import sys
from .general_helpers import *
import subprocess
import numpy as np

# Function to extract the DWI filename
def extract_region_ID(file):
    # Extract the filename
    if os.name == "nt":
        region_name = file.split("\\")[-3]
        if "resized" in region_name:
            region_name = region_name.replace("_resized", "")
    else:
        region_name = file.split("/")[-3]
        if "resized" in region_name:
            region_name = region_name.replace("_resized", "")
    # Return the filename
    return region_name


# Function to extract the BVAL filename
def extract_correct_bval(dwi_file, BMINDS_BVAL_FILES):

    # Extract the correct number for the bval
    if os.name == "nt":
        bval_name = dwi_file.split("\\")[-1]
    else:
        bval_name = dwi_file.split("/")[-1].split(".")[0]
    
    # For all the bvals extracted
    for bval_file in BMINDS_BVAL_FILES:
        # If the bval file has the same name as the bval name (i.e. the number 1000, 3000, etc.)
        if bval_name in bval_file:
            # Return the bval filepath
            return bval_file
    
    # If we don't find the bval file, exit the program
    print("No bval file found for {}".format(dwi_file))
    sys.exit('Exiting program')

# Function to extract the BVEC filename
def extract_correct_bvec(dwi_file, BMINDS_BVEC_FILES):
    # Extract the correct number for the bval
    if os.name == "nt":
        bvec_name = dwi_file.split("\\")[-1]
    else:
        bvec_name = dwi_file.split("/")[-1].split(".")[0]
    
    # For all the bvecs extracted
    for bvec_file in BMINDS_BVEC_FILES:
        # If the bvec file has the same name as the bvec name (i.e. the number 1000, 3000, etc.)
        if bvec_name in bvec_file:
            # Return the bvec filepath
            return bvec_file
    
    # If we don't find the bvec file, exit the program
    print("No bvec file found for {}".format(dwi_file))
    sys.exit('Exiting program')

# Function to determine what type of streamline file it is (dwi, tract-tracing, etc)
def get_streamline_type(file):
    # Extract the filename
    if os.name == "nt":
        streamline_name = file.split("\\")[-1].split(".")[0]
    else:
        streamline_name = file.split("/")[-1].split(".")[0]
    
    # Return different things, depending on the name
    if 'sharp' in streamline_name:
        return 'tracer_tracts_sharp'
    elif 'tracer' in streamline_name:
        return 'tracer_tracts'
    elif 'track' in streamline_name:
        return 'dwi_tracts'
    # If none of the above, return unknown error
    else:
        print("Unknown streamline file type for {}. Name is {}".format(file, streamline_name))
        sys.exit('Exiting program')

# Function to determine what type of injection file it is (density, etc)
def get_injection_type(file):
    # Extract the filename
    if os.name == "nt":
        injection_name = file.split("\\")[-1].split(".")[0]
    else:
        injection_name = file.split("/")[-1].split(".")[0]
    
    # Return different things, depending on the name
    if 'cell_density' in injection_name:
        return 'cell_density'
    elif 'positive_voxels' in injection_name:
        return 'tracer_positive_voxels'
    elif 'signal_normalized' in injection_name:
        return 'tracer_signal_normalized'
    elif 'signal' in injection_name:
        return 'tracer_signal'
    elif 'density' in injection_name:
        return 'streamline_density'
    # If none of the above, return unknown error
    else:
        print("Unknown injection file type for {}. Name is {}".format(file, injection_name))
        sys.exit('Exiting program')

# Function to resize the DWI and extract only the first K shells
def resize_dwi_by_scale(region_item, scale_x=0.5, scale_y=0.5, scale_z=0.5, verbose=False):

    # Define the resized DWI path
    if os.name == "nt":
        resized_name = region_item[0].split("\\")[-1].split(".")[0] + "_resized.nii"
        RESIZED_PATH = os.path.join("\\".join(region_item[0].split("\\")[:-1]), resized_name)
    else:
        resized_name = region_item[0].split("/")[-1].split(".")[0] + "_resized.nii"
        RESIZED_PATH = os.path.join("/".join(region_item[0].split("/")[:-1]), resized_name)
    
    # Check if the paths already exist
    if os.path.exists(RESIZED_PATH):
        if verbose:
            print("Resized DWI path {} already exists. Continuing...".format(RESIZED_PATH))
        return RESIZED_PATH

    # Resize the DWI and save under resized
    MRRESIZE_CMD = "mrgrid {input} regrid -scale {scale_x},{scale_y},{scale_z} {output}".format(input=region_item[0], 
                    scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, output=RESIZED_PATH)
    
    print("Running command: {}".format(MRRESIZE_CMD))
    subprocess.run(MRRESIZE_CMD, shell=True)

    # Return the resized path
    return RESIZED_PATH

# Function to extract the first K shells of the bvals and bvecs
def extract_K_shells_bvals_bvecs(region_item, K=3, verbose=False):

        # Define the resized DWI path
    if os.name == "nt":
        bval_shell_name = region_item[1].split("\\")[-1].split(".")[0] + "_{}_shell_bval.bval".format(K)
        bvec_shell_name = region_item[2].split("\\")[-1].split(".")[0] + "_{}_shell_bvec.bvec".format(K)
        BVAL_SHELL_PATH = os.path.join("\\".join(region_item[1].split("\\")[:-1]), bval_shell_name)
        BVEC_SHELL_PATH = os.path.join("\\".join(region_item[2].split("\\")[:-1]), bvec_shell_name)
    else:
        bval_shell_name = region_item[1].split("/")[-1].split(".")[0] + "_{}_shell_bval.bval".format(K)
        bvec_shell_name = region_item[2].split("/")[-1].split(".")[0] + "_{}_shell_bvec.bvec".format(K)
        BVAL_SHELL_PATH = os.path.join("/".join(region_item[1].split("/")[:-1]), bval_shell_name)
        BVEC_SHELL_PATH = os.path.join("/".join(region_item[2].split("/")[:-1]), bvec_shell_name)

    if os.path.exists(BVAL_SHELL_PATH) and os.path.exists(BVEC_SHELL_PATH):
        if verbose:
            print("Extracted {K}-shell bval {bval} and bvec {bvec} filepath already exist. Continuing...".format(
                                                                    K=K, bval=BVAL_SHELL_PATH, bvec=BVEC_SHELL_PATH))
        return (BVAL_SHELL_PATH, BVEC_SHELL_PATH)
    
    # Extract the first K shells of the bvals and bvecs
    bval_text = np.loadtxt(region_item[1])
    bvec_text = np.loadtxt(region_item[2])

    print("bvec shape: {}".format(bvec_text.shape))

    # Get the first K shells
    bval_shell = bval_text[:K]
    bvec_shell = bvec_text[:][:K]

    # Save the first K shells
    np.savetxt(BVAL_SHELL_PATH, bval_shell, fmt='%i')
    np.savetxt(BVEC_SHELL_PATH, bvec_shell, fmt='%f')

    # Return the first K shells
    return [BVAL_SHELL_PATH, BVEC_SHELL_PATH]

# Join different bval and bvec files for the same region
def join_dwi_diff_bvals_bvecs(DWI_LIST):
    # This stores which regions we've already done this for
    SEEN_REGIONS = []
    # This stores all the concat paths
    ALL_CONCAT_PATHS = []
    RESIZED_ALL_CONCAT_PATHS = []
    
    # For each DWI file
    for dwi in DWI_LIST:

        # Get the region, or common element ID
        region_ID = dwi[0]

        # If we've already done this region, skip it
        if region_ID in SEEN_REGIONS:
            continue

        # Add the region to the seen regions
        SEEN_REGIONS.append(region_ID)

        # Create list of all DWIs, BVALs and BVECs for this same region
        same_region_list = [dwi[1:] for dwi in DWI_LIST if dwi[0] == region_ID]

        # Create list of all resized DWIs, BVALs and BVECs for this same region
        resized_region_list = get_resized_bval_bvec_lists(same_region_list)
        
        # Convert to mif and concatenate for both the normal and resized DWIs
        (CONCAT_NII_PATH, CONCAT_MIF_PATH, CONCAT_BVALS_PATH, 
         CONCAT_BVECS_PATH) = convert_and_concatenate(same_region_list)
        
        (RESIZED_CONCAT_NII_PATH, RESIZED_CONCAT_MIF_PATH, RESIZED_CONCAT_BVALS_PATH, 
         RESIZED_CONCAT_BVECS_PATH) = convert_and_concatenate(resized_region_list)

        # Add the concatenated path to the list
        ALL_CONCAT_PATHS.append([CONCAT_NII_PATH, CONCAT_MIF_PATH, CONCAT_BVALS_PATH, CONCAT_BVECS_PATH])
        RESIZED_ALL_CONCAT_PATHS.append([RESIZED_CONCAT_NII_PATH, RESIZED_CONCAT_MIF_PATH, RESIZED_CONCAT_BVALS_PATH, RESIZED_CONCAT_BVECS_PATH])

    # Return all the concatenated paths
    return (ALL_CONCAT_PATHS, RESIZED_ALL_CONCAT_PATHS)

# Function to get the resized DWI, bval and bvec paths
def get_resized_bval_bvec_lists(same_region_list, verbose=False):
    
    # This stores the resized paths - resets with every new region
    RESIZE_PATHS = []
    FIRST_K_SHELL_BVALS_BVECS = []
    resized_region_list = []

    # Resize and extract the first K shells for all the DWIs
    for region_item in same_region_list:
        # Resize the DWI and extract the first K shells
        RESIZE_PATHS.append(resize_dwi_by_scale(region_item, scale_x=0.5, scale_y=0.5, scale_z=0.5))
        FIRST_K_SHELL_BVALS_BVECS.append(extract_K_shells_bvals_bvecs(region_item, K=3))
        # Create a new list with the resized and first K shells
        resized_region_list.append([RESIZE_PATHS[-1], FIRST_K_SHELL_BVALS_BVECS[-1][0], FIRST_K_SHELL_BVALS_BVECS[-1][1]])

    # Return the resized region list
    return resized_region_list

# Function to do all the conversion and concatenation
def convert_and_concatenate(DWI_LIST):
    
    # This stores the MIF - resets with every new region
    MIF_PATHS = []

    # Convert all to mif using the BVALs and BVECs
    for region_item in DWI_LIST:
        # Get the mif path
        MIF_PATHS.append(convert_to_mif(region_item))
    
    # Create string for what mifs to concatenate
    mif_files_string = " ".join(MIF_PATHS)
    # Get the concatenated path
    CONCAT_MIF_PATH = concatenate_mif(MIF_PATHS, mif_files_string)

    # Convert back to nii
    (CONCAT_NII_PATH, CONCAT_BVALS_PATH, CONCAT_BVECS_PATH) = convert_to_nii(CONCAT_MIF_PATH)

    # Return the concatenated stuff
    return (CONCAT_NII_PATH, CONCAT_MIF_PATH, CONCAT_BVALS_PATH, CONCAT_BVECS_PATH)


# Conversion to MIF
def convert_to_mif(region_item, verbose=False):
    # Create the MIF path
    if os.name == "nt":
        mif_name = region_item[0].split("\\")[-1].split(".")[0] + "_mif.mif"
        MIF_PATH = os.path.join("\\".join(region_item[0].split("\\")[:-1]), mif_name)
    else:
        mif_name = region_item[0].split("/")[-1].split(".")[0] + "_mif.mif"
        MIF_PATH = os.path.join("/".join(region_item[0].split("/")[:-1]), mif_name)
    
    # Check if the MIF path already exists
    if os.path.exists(MIF_PATH):
        if verbose:
            print("MIF path {} already exists. Continuing...".format(MIF_PATH))
        return MIF_PATH
    
    # If it doesn't exist, convert to mif
    MIF_CMD = "mrconvert {input_nii} -fslgrad {bvec} {bval} {output}".format(input_nii=region_item[0], 
                                                                        bval=region_item[1], 
                                                                        bvec=region_item[2], 
                                                                        output=MIF_PATH)
    print("Running command: {}".format(MIF_CMD))
    subprocess.run(MIF_CMD, shell=True)

    # Return the mif path
    return MIF_PATH

# Concatenate MIF
def concatenate_mif(MIF_PATHS, mif_files_string, verbose=False):
    # Define the output concatentated path
    if os.name == "nt":
        CONCAT_FOLDER = os.path.join("\\".join(MIF_PATHS[0].split("\\")[:-2]), "Concatenated_Data")
        check_output_folders(CONCAT_FOLDER, "CONCAT_FOLDER", wipe=False)
        CONCAT_PATH = os.path.join(CONCAT_FOLDER, "DWI_concatenated.mif")
    else:
        CONCAT_FOLDER = os.path.join("/".join(MIF_PATHS[0].split("/")[:-2]), "Concatenated_Data")
        check_output_folders(CONCAT_FOLDER, "CONCAT_FOLDER", wipe=False)
        CONCAT_PATH = os.path.join(CONCAT_FOLDER, "DWI_concatenated.mif")
    
    # Check if the concatenated path already exists
    if os.path.exists(CONCAT_PATH):
        if verbose:
            print("Concatenated path {} already exists. Continuing...".format(CONCAT_PATH))
        return CONCAT_PATH
    
    # Concatenate mifs command
    CONCAT_CMD = "mrcat {inputs} {output}".format(inputs=mif_files_string, output=CONCAT_PATH)
    print("Running command: {}".format(CONCAT_CMD))
    subprocess.run(CONCAT_CMD, shell=True)

    # Return the concatenated path
    return CONCAT_PATH

# Function to convert to nii from MIF
def convert_to_nii(MIF_PATH, verbose=False):
    # Define the output nii, bvals and bvecs path
    NII_PATH = MIF_PATH.replace(".mif", ".nii")
    BVALS_PATH = MIF_PATH.replace(".mif", ".bvals")
    BVECS_PATH = MIF_PATH.replace(".mif", ".bvecs")

    # Check if it already exists - if it does, return it
    if os.path.exists(NII_PATH):
        if verbose:
            print("NII path {} already exists. Continuing...".format(NII_PATH))
        return (NII_PATH, BVALS_PATH, BVECS_PATH)

    # Define the conversion command
    CONVERT_BACK_CMD = "mrconvert {input_mif} {output_nii} -export_grad_fsl {bvecs_path} {bvals_path}".format(
                        input_mif=MIF_PATH, output_nii=NII_PATH, bvecs_path=BVECS_PATH, bvals_path=BVALS_PATH)
    print("Running command: {}".format(CONVERT_BACK_CMD))    
    subprocess.run(CONVERT_BACK_CMD, shell=True)

    # Return the nii path
    return (NII_PATH, BVALS_PATH, BVECS_PATH)

# Function to get the selected items from a list
def extract_from_input_list(GENERAL_FILES, ITEMS_NEEDED, list_type):
    
    # Create dictionary that defines what to get
    items_to_get = {}

    # Check whether we're passing in a list or a string
    if isinstance(ITEMS_NEEDED, str):
        ITEMS_NEEDED = [ITEMS_NEEDED]
    if isinstance(GENERAL_FILES, str):
        GENERAL_FILES = [GENERAL_FILES]

    # Extract things differently, depending on the list type being passed
    if list_type == "dwi":
        # Define indices
        DWI_PATH_NII = 0
        DWI_PATH_MIF = 1
        BVAL_PATH = 2
        BVEC_PATH = 3

        # For every item in items, get the item
        for item in ITEMS_NEEDED:
            if item == "dwi_nii":
                items_to_get["dwi_nii"] = GENERAL_FILES[DWI_PATH_NII]
            elif item == "dwi_mif":
                items_to_get["dwi_mif"] = GENERAL_FILES[DWI_PATH_MIF]
            elif item == "bval":
                items_to_get["bval"] = GENERAL_FILES[BVAL_PATH]
            elif item == "bvec":
                items_to_get["bvec"] = GENERAL_FILES[BVEC_PATH]
            else:
                print("Item {} of DWI not found".format(item))
                sys.exit('Exiting program')
    
    # Slightly different with streamlines - here we have a list of lists, and
    # we're not necessarily sure in what order it appends the files
    elif list_type == "streamline":
        # Define indices
        STREAMLINE_TYPE = 0
        STREAMLINE_PATH = 1

        # For every item in items, get the item
        for item in ITEMS_NEEDED:
            # Find the streamline file that has the type we want
            if item == "tracer_tracts_sharp":
                items_to_get["tracer_tracts_sharp"] = [streamline_list[STREAMLINE_PATH] for streamline_list in GENERAL_FILES if streamline_list[STREAMLINE_TYPE] == "tracer_tracts_sharp"]
            elif item == "dwi_tracts":
                items_to_get["dwi_tracts"] = [streamline_list[STREAMLINE_PATH] for streamline_list in GENERAL_FILES if streamline_list[STREAMLINE_TYPE] == "dwi_tracts"]
            elif item == "tracer_tracts":
                items_to_get["tracer_tracts"] = [streamline_list[STREAMLINE_PATH] for streamline_list in GENERAL_FILES if streamline_list[STREAMLINE_TYPE] == "tracer_tracts"]
            else:
                print("Item {} of streamlines not found".format(item))
                sys.exit('Exiting program')
    
    # The same as above is done for injections
    elif list_type == "injection":
        # Define indices
        INJECTION_TYPE = 0
        INJECTION_PATH = 1

        # For every item in items, get the item
        for item in ITEMS_NEEDED:
            # Find the injection file that has the type we want
            if item == "tracer_signal_normalized":
                items_to_get["tracer_signal_normalized"] = [injection_list[INJECTION_PATH] for injection_list in GENERAL_FILES if injection_list[INJECTION_TYPE] == "tracer_signal_normalized"]
            elif item == "tracer_positive_voxels":
                items_to_get["tracer_positive_voxels"] = [injection_list[INJECTION_PATH] for injection_list in GENERAL_FILES if injection_list[INJECTION_TYPE] == "tracer_positive_voxels"]
            elif item == "cell_density":
                items_to_get["cell_density"] = [injection_list[INJECTION_PATH] for injection_list in GENERAL_FILES if injection_list[INJECTION_TYPE] == "cell_density"]
            elif item == "streamline_density":
                items_to_get["streamline_density"] = [injection_list[INJECTION_PATH] for injection_list in GENERAL_FILES if injection_list[INJECTION_TYPE] == "streamline_density"]
            elif item == "tracer_signal":
                items_to_get["tracer_signal"] = [injection_list[INJECTION_PATH] for injection_list in GENERAL_FILES if injection_list[INJECTION_TYPE] == "tracer_signal"]
            else:
                print("Item {} of injections not found".format(item))
                sys.exit('Exiting program')
    
    # For atlas and STPT, it's just the index
    elif list_type == "atlas_stpt":
        ATLAS_PATH = 0
        ATLAS_LABEL_PATH = 1
        STPT_PATH = 2

        # For every item in items, get the item
        for item in ITEMS_NEEDED:
            if item == "atlas":
                items_to_get["atlas"] = GENERAL_FILES[ATLAS_PATH]
            elif item == "atlas_label":
                items_to_get["atlas_label"] = GENERAL_FILES[ATLAS_LABEL_PATH]
            elif item == "stpt":
                items_to_get["stpt"] = GENERAL_FILES[STPT_PATH]
            else:
                print("Item {} of atlas and STPT not found".format(item))
                sys.exit('Exiting program')
    
    # For transforms, it's also just the index
    elif list_type == "transforms":
        # For every item in items, get the item
        for item in ITEMS_NEEDED:
            if item == "mbca_transform":
                items_to_get["mbca_transform"] = GENERAL_FILES[0]
            else:
                print("Item {} of transforms not found".format(item))
                sys.exit('Exiting program')
    
    # If not any of the above, exit the program
    else:
        print("List type {} not found".format(list_type))
        sys.exit('Exiting program')
            
    return items_to_get

# Create initial lists for the above function
def create_initial_lists(BMINDS_UNZIPPED_DWI_FILES, BMINDS_BVAL_FILES, BMINDS_BVEC_FILES, BMINDS_STREAMLINE_FILES,
                            BMINDS_INJECTION_FILES):
    DWI_LIST = []
    STREAMLINE_LIST = []
    INJECTION_LIST = []

    # For each DWI file
    for dwi_file in BMINDS_UNZIPPED_DWI_FILES:

        # Check that it's not a concat or resized file - skip if it is
        if "concat" in dwi_file or "resized" in dwi_file:
            continue

        # Get the region ID
        region_ID = extract_region_ID(dwi_file)

        # Get the bval and bvec files
        bval_path = extract_correct_bval(dwi_file, BMINDS_BVAL_FILES)
        bvec_path = extract_correct_bvec(dwi_file, BMINDS_BVEC_FILES)

        # Append to a DWI list
        DWI_LIST.append([region_ID, dwi_file, bval_path, bvec_path])

    # For each streamline file
    for streamline_file in BMINDS_STREAMLINE_FILES:
        # Get the region ID
        region_ID = extract_region_ID(streamline_file)
        # Get the type of streamline file it is
        streamline_type = get_streamline_type(streamline_file)
        # Append all the data to the dictionary
        STREAMLINE_LIST.append([region_ID, streamline_type, streamline_file])

    # For each injection file
    for injection_file in BMINDS_INJECTION_FILES:
        # Ignore the ones with small in the filename
        if "small" in injection_file:
            continue
        # Get the region ID
        region_ID = extract_region_ID(injection_file)
        # Get the type of injection file it is
        injection_type = get_injection_type(injection_file)
        # Append all the data to the dictionary
        INJECTION_LIST.append([region_ID, injection_type, injection_file])

    # Return the lists
    return (DWI_LIST, STREAMLINE_LIST, INJECTION_LIST)