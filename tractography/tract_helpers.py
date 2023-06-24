import os
import sys
import shutil
import glob
import numpy as np

# -------------------------------------------------- MAIN FUNCTION MODULES -------------------------------------------------- #
def get_main_paths(hpc):
    # Depending on whether we're in HPC or not, paths change
    if hpc == True:
        DWI_MAIN_FOLDER = "/rds/general/user/hsa22/home/dissertation/tractography"
        DWI_OUTPUT_FOLDER = "/rds/general/user/hsa22/home/dissertation/tractography/dsi_outputs"

        DSI_COMMAND = "singularity exec dsistudio_latest.sif dsi_studio"

        ATLAS_FOLDER = "/home/hsa22/ConnectomePreprocessing/atlas"
        TRACT_FOLDER = "/home/hsa22/ConnectomePreprocessing/tracts"
    else:
        # Define paths based on whether we're Windows or Linux
        if os.name == "nt":
            ALL_DATA_FOLDER = os.path.realpath(r"C:\\tractography\\data")
            SUBJECTS_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "subjects"))
            DWI_OUTPUT_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "dsi_outputs"))
            DWI_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "dwi"))
            T1_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "t1"))
            FMRI_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "fmri"))

            DSI_COMMAND = "dsi_studio"

            ATLAS_FOLDER = os.path.realpath(r"C:\\tractography\\data\\atlas")
        else:
            ALL_DATA_FOLDER = os.path.realpath(os.path.join(os.getcwd(), "data"))
            SUBJECTS_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "subjects"))
            DWI_OUTPUT_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "dsi_outputs"))
            DWI_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "dwi"))
            T1_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "t1"))
            FMRI_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "fmri"))

            DSI_COMMAND = "dsi_studio"

            ATLAS_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "atlas"))

    MAIN_STUDIO_PATH = os.path.join(DWI_OUTPUT_FOLDER, "studio")
    MAIN_MRTRIX_PATH = os.path.join(DWI_OUTPUT_FOLDER, "mrtrix")
    MAIN_FSL_PATH = os.path.join(DWI_OUTPUT_FOLDER, "fsl")

    # Return folder names
    return (ALL_DATA_FOLDER, SUBJECTS_FOLDER, DWI_OUTPUT_FOLDER, DWI_MAIN_FOLDER, 
            T1_MAIN_FOLDER, FMRI_MAIN_FOLDER, DSI_COMMAND, ATLAS_FOLDER, MAIN_STUDIO_PATH, 
            MAIN_MRTRIX_PATH, MAIN_FSL_PATH)

# Check that input folders exist
def check_input_folders(folder, name):
    if not os.path.exists(folder):
        print("--- {} folder not found. Please add folder: {}".format(name, folder))
        sys.exit('Exiting program')
    else:
        print("--- {} folder found. Continuing...".format(name))

# Check that output folders with subfolders are in suitable shape
def check_output_folders_with_subfolders(folder, name):
    if not os.path.exists(folder):
        print("--- {} folder not found. Created folder: {}".format(name, folder))
        os.makedirs(folder)
    # If it has content, delete it
    else:
        print("--- {} folder found. Continuing...".format(name))
        if len(os.listdir(folder)) != 0:
            print("{} folder has content. Deleting content...".format(name))
            # Since this can have both folders and files, we need to check if it's a file or folder to remove
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

# Check that output folders are in suitable shape
def check_output_folders(folder, name):
    if not os.path.exists(folder):
        print("--- {} folder not found. Created folder: {}".format(name, folder))
        os.makedirs(folder)
    # If it has content, delete it
    else:
        print("--- {} folder found. Continuing...".format(name))
        if len(os.listdir(folder)) != 0:
            print("{} folder has content. Deleting content...".format(name))
            for file in glob.glob(os.path.join(folder, "*")):
                os.remove(file)
            print("Content deleted. Continuing...")

# Function to check output folders without wiping
def check_output_folders_nowipe(folder, name):
    if not os.path.exists(folder):
        print("--- {} folder not found. Created folder: {}".format(name, folder))
        os.makedirs(folder)

# Retrieve (GLOB) files
def glob_files(PATH_NAME, file_format):
    INPUT_FILES = []
    for file in glob.glob(os.path.join(PATH_NAME, os.path.join("**", "*.{}".format(file_format))), recursive=True):
        INPUT_FILES.append(file)
    return INPUT_FILES

# Check that the retrieved (GLOBBED) files are not empty
def check_globbed_files(files, name):
    if len(files) == 0:
        print("No {} files found. Please add {} files".format(name, name))
        sys.exit('Exiting program')
    else:
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
def extract_t1_fmri_filename(t1_fmri_file):
    # Extract the filename
    filename = t1_fmri_file.split("/")[-1].split("_")[1].split("-")[0]
    # Return the filename
    return filename

# Create a list that associates each subject with its T1 and DWI files
def create_subject_list(DWI_INPUT_FILES, DWI_JSON_HEADERS, B_VAL_FILES, B_VEC_FILES, T1_INPUT_FILES, FMRI_INPUT_FILES):
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
        subject_ID = extract_t1_fmri_filename(t1_file)
        # Append to a T1 list
        T1_LIST.append([subject_ID, t1_file])
    # For each fMRI file
    for fmri_file in FMRI_INPUT_FILES:
        # Get the filename
        subject_ID = extract_t1_fmri_filename(fmri_file)
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
        # Append the subject name, dwi, json, bval, bvec, t1 and fMRI to the list
        SUBJECT_LISTS.append([dwi_name, dwi[1], json[0], bval[0], bvec[0], t1[0], fmri[0]])
        
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