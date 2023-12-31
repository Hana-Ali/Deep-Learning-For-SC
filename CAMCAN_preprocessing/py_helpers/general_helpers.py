import os
import sys
import shutil
import glob
import numpy as np

# -------------------------------------------------- MAIN FUNCTION MODULES -------------------------------------------------- #

def get_main_paths(hpc):
    # Depending on whether we're in HPC or not, paths change
    if hpc == True:
        ALL_DATA_FOLDER = "/rds/general/user/hsa22/ephemeral/CAMCAN"
        SUBJECTS_FOLDER = "" # Empty in the case of HPC
        TRACTOGRAPHY_OUTPUT_FOLDER = os.path.join(ALL_DATA_FOLDER, "dMRI_outputs")
        NIPYPE_OUTPUT_FOLDER = os.path.join(ALL_DATA_FOLDER, "Nipype_outputs")
        # FMRI_MAIN_FOLDER = os.path.join(ALL_DATA_FOLDER, "camcan_parcellated_acompcor/glasser360/fmri700/rest")
        FMRI_MAIN_FOLDER = os.path.join(ALL_DATA_FOLDER, "camcan_parcellated_acompcor")
        ATLAS_FOLDER = os.path.join(ALL_DATA_FOLDER, "Atlas")
        REGISTERED_ATLASES_FOLDER = os.path.join(ATLAS_FOLDER, "registered_atlases")

        PEDRO_MAIN_FOLDER = "/rds/general/user/pam213/home/Data/CAMCAN/"
        DWI_MAIN_FOLDER = os.path.join(PEDRO_MAIN_FOLDER, "dwi")
        T1_MAIN_FOLDER = os.path.join(PEDRO_MAIN_FOLDER, "aamod_dartel_norm_write_00001")
        
        DSI_COMMAND = "singularity exec dsistudio_latest.sif dsi_studio"

    else:
        # Define paths based on whether we're Windows or Linux
        if os.name == "nt":
            ALL_DATA_FOLDER = os.path.realpath(r"C:\\tractography\\data")
            
            SUBJECTS_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "subjects"))
            TRACTOGRAPHY_OUTPUT_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "dsi_outputs"))
            NIPYPE_OUTPUT_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "nipype_outputs"))
            DWI_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "dwi"))
            T1_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "t1"))
            FMRI_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "fmri"))

            DSI_COMMAND = "dsi_studio"

            ATLAS_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "atlas"))

        else:
            ALL_DATA_FOLDER = os.path.realpath(os.path.join(os.getcwd(), "data"))
            SUBJECTS_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "subjects"))
            TRACTOGRAPHY_OUTPUT_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "dsi_outputs"))
            NIPYPE_OUTPUT_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "nipype_outputs"))
            DWI_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "dwi"))
            T1_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "t1"))
            FMRI_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "fmri"))

            DSI_COMMAND = "dsi_studio"

            ATLAS_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "atlas"))

    MAIN_STUDIO_PATH = os.path.join(TRACTOGRAPHY_OUTPUT_FOLDER, "studio")
    MAIN_MRTRIX_PATH = os.path.join(TRACTOGRAPHY_OUTPUT_FOLDER, "mrtrix")
    MAIN_FSL_PATH = os.path.join(TRACTOGRAPHY_OUTPUT_FOLDER, "fsl")


    # Return folder names
    return (ALL_DATA_FOLDER, SUBJECTS_FOLDER, TRACTOGRAPHY_OUTPUT_FOLDER, NIPYPE_OUTPUT_FOLDER, 
            DWI_MAIN_FOLDER, T1_MAIN_FOLDER, FMRI_MAIN_FOLDER, DSI_COMMAND, ATLAS_FOLDER, 
            REGISTERED_ATLASES_FOLDER, MAIN_STUDIO_PATH, MAIN_MRTRIX_PATH, MAIN_FSL_PATH)

# Check that input folders exist
def check_input_folders(folder, name, verbose=False):
    if not os.path.exists(folder):
        print("--- {} folder not found. Please add folder: {}".format(name, folder))
        sys.exit('Exiting program')
    else:
        if verbose:
            print("--- {} folder found. Continuing...".format(name))

# Check that output folders are in suitable shape
def check_output_folders(folder, name, wipe=False, verbose=False):
    if not os.path.exists(folder):
        if verbose:
            print("--- {} folder not found. Created folder: {}".format(name, folder))
        os.makedirs(folder)
    # If it has no content, either continue or delete, depending on wipe
    else:
        # If it has content, delete it
        if wipe:
            if verbose:
                print("--- {} folder found. Wiping...".format(name))
            # If the folder has content, delete it
            if len(os.listdir(folder)) != 0:
                if verbose:
                    print("{} folder has content. Deleting content...".format(name))
                # Since this can have both folders and files, we need to check if it's a file or folder to remove
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                if verbose:
                    print("Content deleted. Continuing...")
        else:
            if verbose:
                print("--- {} folder found. Continuing without wipe...".format(name))
            
# Retrieve (GLOB) files
def glob_files(PATH_NAME, file_format):
    INPUT_FILES = []
    for file in glob.glob(os.path.join(PATH_NAME, os.path.join("**", "*.{}".format(file_format))), recursive=True):
        INPUT_FILES.append(file)
    return INPUT_FILES

# Check that the retrieved (GLOBBED) files are not empty
def check_globbed_files(files, name, verbose=False):
    if len(files) == 0:
        print("No {} files found. Please add {} files".format(name, name))
        sys.exit('Exiting program')
    else:
        if verbose:
            print("{} files found. Continuing...".format(name))

# Function to filter the subjects list based on quality checks done by Pedro
def filter_subjects_list(SUBJECT_LISTS, SUBJECTS_DATA_PATH):
    # Get the filtering data
    FILTERING_DATA_PATH = os.path.join(SUBJECTS_DATA_PATH, "good_subjects.csv")
    # Read the filtering data and choose which to remove
    FILTERING_DATA = np.genfromtxt(FILTERING_DATA_PATH, delimiter=",", dtype=str)
    FILTERING_DATA_REMOVE = FILTERING_DATA[FILTERING_DATA[:, 1] == '0']

    # Get the subjects that passed the quality check
    FILTERED_SUBJECTS = [subject for subject in SUBJECT_LISTS if subject[0] not in FILTERING_DATA_REMOVE[:, 0]]
    
    print("Number of subjects after filtering: {}".format(len(FILTERED_SUBJECTS)))

    return FILTERED_SUBJECTS

# Function to extract the DWI filename
def extract_dwi_filename(dwi_file):
    # Extract the filename
    if os.name == "nt":
        dwi_filename = dwi_file.split("\\sub-")[-1].split("_")[0]
    else:
        dwi_filename = dwi_file.split("/sub-")[-1].split("_")[0]
    # Return the filename
    return dwi_filename

# Function to extract the T1 filename
def extract_t1_filename(t1_file):
    # Extract the filename
    if os.name == "nt":
        t1_filename = t1_file.split("\\")[-1].split("_")[1].split("-")[0]
    else:
        t1_filename = t1_file.split("/")[-1].split("_")[1].split("-")[0]
    # Return the filename
    return t1_filename

# Function to extract the fMRI filename
def extract_fmri_filename(fmri_file):
    # Extract the filename
    if os.name == "nt":
        fmri_filename = fmri_file.split("\\")[-1].split("_")[0]
    else:
        fmri_filename = fmri_file.split("/")[-1].split("_")[0]
    # Return the filename
    return fmri_filename

# Create a list that associates each subject with its T1 and DWI files
def create_subject_list(DWI_INPUT_FILES, DWI_JSON_HEADERS, B_VAL_FILES, B_VEC_FILES, T1_INPUT_FILES, FMRI_INPUT_FILES,
                        verbose=False):
    SUBJECT_LISTS = []
    DWI_LIST = []
    JSON_LIST = []
    B_VAL_LIST = []
    B_VEC_LIST = []
    T1_LIST = []
    FMRI_LIST = []

    # For each DWI file
    for dwi_file in DWI_INPUT_FILES:
        # Get the filename
        subject_ID = extract_dwi_filename(dwi_file)
        # Append to a DWI list
        DWI_LIST.append([subject_ID, dwi_file])
    # For each JSON file
    for json_file in DWI_JSON_HEADERS:
        # Get the filename
        subject_ID = extract_dwi_filename(json_file)
        # Append to a JSON list
        JSON_LIST.append([subject_ID, json_file])
    # For each BVAL file
    for bval_file in B_VAL_FILES:
        # Get the filename
        subject_ID = extract_dwi_filename(bval_file)
        # Append to a BVAL list
        B_VAL_LIST.append([subject_ID, bval_file])
    # For each BVEC file
    for bvec_file in B_VEC_FILES:
        # Get the filename
        subject_ID = extract_dwi_filename(bvec_file)
        # Append to a BVEC list
        B_VEC_LIST.append([subject_ID, bvec_file])
    # For each T1 file
    for t1_file in T1_INPUT_FILES:
        # Get the filename
        subject_ID = extract_t1_filename(t1_file)
        # Append to a T1 list
        T1_LIST.append([subject_ID, t1_file])
    # For each fMRI file
    for fmri_file in FMRI_INPUT_FILES:
        # Get the filename
        subject_ID = extract_fmri_filename(fmri_file)
        # Append to a fMRI list
        FMRI_LIST.append([subject_ID, fmri_file])

    # Join the two lists based on common subject name
    for dwi in DWI_LIST:
        # Get the name, or common element ID
        dwi_name = dwi[0]
        # Based on this name, get every json, bval, bvec and t1 that has the same name
        json = [json[1] for json in JSON_LIST if json[0] == dwi_name]
        bval = [bval[1] for bval in B_VAL_LIST if bval[0] == dwi_name]
        bvec = [bvec[1] for bvec in B_VEC_LIST if bvec[0] == dwi_name]
        t1 = [t1[1] for t1 in T1_LIST if t1[0] == dwi_name]
        fmri = [fmri[1] for fmri in FMRI_LIST if fmri[0] == dwi_name]

        # CHECK THAT SUBJECT HAS ALL FILES
        if not len(json):
            if verbose:
                print("Subject {} does not have JSON file. Not appending to subjects".format(dwi_name))
            continue
        if not len(bval):
            if verbose:
                print("Subject {} does not have BVAL file. Not appending to subjects".format(dwi_name))
            continue
        if not len(bvec):
            if verbose:
                print("Subject {} does not have BVEC file. Not appending to subjects".format(dwi_name))
            continue
        if not len(t1):
            if verbose:
                print("Subject {} does not have T1 file. Not appending to subjects".format(dwi_name))
            continue
        if not len(fmri):
            if verbose:
                print("Subject {} does not have fMRI file. Not appending to subjects".format(dwi_name))
            continue

        # Append the subject name, dwi, json, bval, bvec, t1 and fMRI to the list
        SUBJECT_LISTS.append([dwi_name, dwi[1], json[0], bval[0], bvec[0], t1[0], fmri])
    
    # print("Number of subjects: {}".format(len(SUBJECT_LISTS)))
        
    return SUBJECT_LISTS

# Get filename
def get_filename(SUBJECT_FILES, items):
    # Define the indices of the subject list
    SUBJECT_DWI_NAME = 0
    SUBJECT_DWI_INDEX = 1
    SUBJECT_DWI_JSON_INDEX = 2
    SUBJECT_BVAL_INDEX = 3
    SUBJECT_BVEC_INDEX = 4
    SUBJECT_T1_INDEX = 5
    SUBJECT_FMRI_INDEX = 6
    # Create dictionary that defines what to get
    items_to_get = {}

    # Check whether we're passing in a list or a string
    if isinstance(items, str):
        items = [items]

    # For every item in items, get the item
    for item in items:
        if item == "filename":
            items_to_get["filename"] = SUBJECT_FILES[SUBJECT_DWI_NAME]
        elif item == "dwi":
            items_to_get["dwi"] = SUBJECT_FILES[SUBJECT_DWI_INDEX]
        elif item == "json":
            items_to_get["json"] = SUBJECT_FILES[SUBJECT_DWI_JSON_INDEX]
        elif item == "bval":
            items_to_get["bval"] = SUBJECT_FILES[SUBJECT_BVAL_INDEX]
        elif item == "bvec":
            items_to_get["bvec"] = SUBJECT_FILES[SUBJECT_BVEC_INDEX]
        elif item == "t1":
            items_to_get["t1"] = SUBJECT_FILES[SUBJECT_T1_INDEX]
        elif item == "fmri":
            items_to_get["fmri"] = SUBJECT_FILES[SUBJECT_FMRI_INDEX]
        else:
            print("Item {} not found".format(item))
            sys.exit('Exiting program')
            
    return items_to_get

# Function to get the fmri files that have atlases
def get_fmri_for_atlas(FMRI_MAIN_FOLDER, ATLAS_FILES):

    # Get the folder names in the fmri folder
    FMRI_FOLDER_NAMES = os.listdir(FMRI_MAIN_FOLDER)

    # Get the atlases common with the fmri folder names
    COMMON_ATLAS_FILES = [atlas_file for atlas_file in ATLAS_FILES if any(fmri_folder in atlas_file for fmri_folder in FMRI_FOLDER_NAMES)]

    # Get the fmri folder names that have atlases
    FMRI_FOLDER_NAMES = [fmri_folder for fmri_folder in FMRI_FOLDER_NAMES if any(fmri_folder in atlas_file for atlas_file in COMMON_ATLAS_FILES)]

    # Get the fmri files for each folder
    FMRI_INPUT_FILES = []
    for fmri_folder in FMRI_FOLDER_NAMES:
        # Get the fmri files for each folder
        FMRI_INPUT_FILES.append(glob_files(os.path.join(FMRI_MAIN_FOLDER, fmri_folder), "mat"))

    # Flatten the list
    FMRI_INPUT_FILES = [item for sublist in FMRI_INPUT_FILES for item in sublist]

    # Return the fmri files
    return (FMRI_INPUT_FILES, COMMON_ATLAS_FILES)

# Function to get the atlas choice
def get_atlas_choice(FILTERED_SUBJECT_LIST, COMMON_ATLAS_FILES, REGISTERED_ATLASES_FOLDER):
    
    print("length of filtered subject list: {}".format(len(FILTERED_SUBJECT_LIST)))

    # Get all the atlases in the registered atlas folder
    REGISTERED_ATLASES = os.listdir(REGISTERED_ATLASES_FOLDER)

    # Get the atlas names in common and registered atlases
    ATLAS_NAMES = [atlas_file.split(os.sep)[-1].split(".")[0] for atlas_file in COMMON_ATLAS_FILES]
    REGISTERED_ATLAS_NAMES = [atlas_file.split(os.sep)[-1] for atlas_file in REGISTERED_ATLASES]

    print("atlas names: {}".format(ATLAS_NAMES))

    # Get the atlas names that are not in the registered atlases
    UNREGISTERED_ATLAS_NAMES = [atlas_name for atlas_name in ATLAS_NAMES if atlas_name not in REGISTERED_ATLAS_NAMES]
    # Remove the jubrain atlas
    UNREGISTERED_ATLAS_NAMES = [atlas_name for atlas_name in UNREGISTERED_ATLAS_NAMES if atlas_name != "jubrain"]

    # If there are no unregistered atlases, exit the program
    if len(UNREGISTERED_ATLAS_NAMES) == 0:
        print("No unregistered atlases found. Please add unregistered atlases")
        sys.exit('Exiting program')

    # Choose a random unregistered atlas
    ATLAS_CHOICE = np.random.choice(UNREGISTERED_ATLAS_NAMES)

    # New subject list
    NEW_SUBJECT_LIST = []


    # For each subject
    for subject in FILTERED_SUBJECT_LIST:

        # Get the fmri files for each subject
        FMRI_FILES = get_filename(subject, "fmri")["fmri"]

        # Get the fmri files that have the atlas choice
        FMRI_FILES_WITH_ATLAS = [fmri_file.lower() for fmri_file in FMRI_FILES if ATLAS_CHOICE.lower() in fmri_file]

        # If there are no fmri files with atlases, exit the program
        if len(FMRI_FILES_WITH_ATLAS) == 0:
            print("No fmri files with atlases found. Please add fmri files with atlases")
            sys.exit('Exiting program')

        # Get the atlas path for the atlas choice
        ATLAS_PATH = [atlas_file for atlas_file in COMMON_ATLAS_FILES if ATLAS_CHOICE in atlas_file][0]

        # Add everything from the old list to the new subject list, and append the correct fmri files
        NEW_SUBJECT_LIST.append(subject[:6] + FMRI_FILES_WITH_ATLAS)
    
    # Return the fmri files with atlas and the atlas path
    return (ATLAS_PATH, NEW_SUBJECT_LIST)