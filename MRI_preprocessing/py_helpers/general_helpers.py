import os
import sys
import shutil
import glob
import numpy as np
import regex as re
import nipype.interfaces.io as nio
from nipype import Node
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline.engine import Workflow

# -------------------------------------------------- MAIN FUNCTION MODULES -------------------------------------------------- #
def get_nipype_datasource(hpc):

    # Get paths, depending on whether we're in HPC or not
    hpc = False
    (ALL_DATA_FOLDER, SUBJECTS_FOLDER, TRACTOGRAPHY_OUTPUT_FOLDER, NIPYPE_OUTPUT_FOLDER, 
        DWI_MAIN_FOLDER, T1_MAIN_FOLDER, FMRI_MAIN_FOLDER, DSI_COMMAND, ATLAS_FOLDER, 
            MAIN_STUDIO_PATH, MAIN_MRTRIX_PATH, MAIN_FSL_PATH) = get_main_paths(hpc)
    
    # Get the subject IDs
    SUBJECT_IDS = os.listdir(FMRI_MAIN_FOLDER)

    # Create iterable list of subjects
    info_source = Node(IdentityInterface(fields=['subject_id']), name='info_source')
    info_source.iterables = [('subject_id', SUBJECT_IDS)]

    # Grab the data
    dwi_files = os.path.join(DWI_MAIN_FOLDER, 'sub-{subject_id}', 'dwi', '*_dwi.nii.gz')
    fmri_files = os.path.join(FMRI_MAIN_FOLDER, '{subject_id}', 'Rest', '*.nii')
    t1_files = os.path.join(T1_MAIN_FOLDER, '{subject_id}', 'structurals', '*.nii')

    # Create templates and grab the files using SelectFiles
    templates = {'dwi': dwi_files, 'fmri': fmri_files, 't1': t1_files}
    select_files = Node(nio.SelectFiles(templates, 
                                        base_directory=SUBJECTS_FOLDER),
                        name='select_files')

    # Return the info source, select files and nipype output folder
    return [info_source, select_files, NIPYPE_OUTPUT_FOLDER]


def get_dmri_fmri_arguments(hpc):

    # Get paths, depending on whether we're in HPC or not
    hpc = False
    (ALL_DATA_FOLDER, SUBJECTS_FOLDER, TRACTOGRAPHY_OUTPUT_FOLDER, NIPYPE_OUTPUT_FOLDER, 
        DWI_MAIN_FOLDER, T1_MAIN_FOLDER, FMRI_MAIN_FOLDER, DSI_COMMAND, ATLAS_FOLDER, 
            MAIN_STUDIO_PATH, MAIN_MRTRIX_PATH, MAIN_FSL_PATH) = get_main_paths(hpc)

    # Check if input folders - if not, exit program
    check_input_folders(DWI_MAIN_FOLDER, "DWI")
    check_input_folders(T1_MAIN_FOLDER, "T1")
    check_input_folders(FMRI_MAIN_FOLDER, "fMRI")
    check_input_folders(ATLAS_FOLDER, "Atlas")

    # If output folderes don't exist, create them
    check_output_folders_with_subfolders(TRACTOGRAPHY_OUTPUT_FOLDER, "Tractography output")
    check_output_folders(MAIN_STUDIO_PATH, "Studio", wipe=True)
    check_output_folders(MAIN_MRTRIX_PATH, "MRtrix", wipe=True)
    check_output_folders(MAIN_FSL_PATH, "FSL", wipe=True)
        
    # --------------- Get DWI, BVAL, BVEC, T1, fMRI from subdirectories --------------- #
    DWI_INPUT_FILES = glob_files(DWI_MAIN_FOLDER, "nii.gz")
    DWI_JSON_HEADERS = glob_files(DWI_MAIN_FOLDER, "json")
    B_VAL_FILES = glob_files(DWI_MAIN_FOLDER, "bval")
    B_VEC_FILES = glob_files(DWI_MAIN_FOLDER, "bvec")
    T1_INPUT_FILES = glob_files(T1_MAIN_FOLDER, "nii")
    FMRI_INPUT_FILES = glob_files(FMRI_MAIN_FOLDER, "mat")
    ATLAS_FILES = glob_files(ATLAS_FOLDER, "nii.gz")
    
    # Clean up T1 template files
    T1_INPUT_FILES = list(filter(lambda x: not re.search('Template', x), T1_INPUT_FILES))
    
    # If no files are found - exit the program
    check_globbed_files(DWI_INPUT_FILES, "DWI")
    check_globbed_files(DWI_JSON_HEADERS, "JSON")
    check_globbed_files(B_VAL_FILES, "BVAL")
    check_globbed_files(B_VEC_FILES, "BVEC")
    check_globbed_files(T1_INPUT_FILES, "T1")
    check_globbed_files(FMRI_INPUT_FILES, "fMRI")
    check_globbed_files(ATLAS_FILES, "Atlas")

    # --------------- Create list of all data for each subject and filter --------------- #
    SUBJECT_LISTS = create_subject_list(DWI_INPUT_FILES, DWI_JSON_HEADERS, B_VAL_FILES, 
                                        B_VEC_FILES, T1_INPUT_FILES, FMRI_INPUT_FILES)
    # Depending on HPC, the filter CSV file is in a different location
    if hpc:
        FILTERED_SUBJECT_LIST = filter_subjects_list(SUBJECT_LISTS, ALL_DATA_FOLDER)
    else:
        FILTERED_SUBJECT_LIST = filter_subjects_list(SUBJECT_LISTS, SUBJECTS_FOLDER)

    # --------------- Get the correct atlas depending on parcellated fMRI --------------- #
    ATLAS_CHOSEN = get_atlas_choice(FILTERED_SUBJECT_LIST, ATLAS_FILES)

    
    # --------------- Defining inputs for mapping parallel --------------- #
    mapping_inputs = list(zip(FILTERED_SUBJECT_LIST, [ATLAS_CHOSEN]*len(FILTERED_SUBJECT_LIST), 
                              [MAIN_STUDIO_PATH]*len(FILTERED_SUBJECT_LIST), [MAIN_MRTRIX_PATH]*len(FILTERED_SUBJECT_LIST), 
                              [MAIN_FSL_PATH]*len(FILTERED_SUBJECT_LIST), [DSI_COMMAND]*len(FILTERED_SUBJECT_LIST)))

    # Return the mapping inputs
    return mapping_inputs


def get_main_paths(hpc):
    # Depending on whether we're in HPC or not, paths change
    if hpc == True:
        ALL_DATA_FOLDER = "/rds/general/user/hsa22/home/dissertation"
        SUBJECTS_FOLDER = "" # Empty in the case of HPC
        TRACTOGRAPHY_OUTPUT_FOLDER = os.path.join(ALL_DATA_FOLDER, "output_data")
        NIPYPE_OUTPUT_FOLDER = os.path.join(ALL_DATA_FOLDER, "nipype_outputs")
        FMRI_MAIN_FOLDER = os.path.join(ALL_DATA_FOLDER, "camcan_parcellated_acompcor/schaefer232/fmri700/rest")
        ATLAS_FOLDER = os.path.join(ALL_DATA_FOLDER, "atlas")

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
            MAIN_STUDIO_PATH, MAIN_MRTRIX_PATH, MAIN_FSL_PATH)

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
def check_output_folders(folder, name, wipe=True):
    if not os.path.exists(folder):
        print("--- {} folder not found. Created folder: {}".format(name, folder))
        os.makedirs(folder)
    # If it has no content, either continue or delete, depending on wipe
    else:
        # If it has content, delete it
        if wipe:
            print("--- {} folder found. Wiping...".format(name))
            if len(os.listdir(folder)) != 0:
                print("{} folder has content. Deleting content...".format(name))
                for file in glob.glob(os.path.join(folder, "*")):
                    os.remove(file)
                print("Content deleted. Continuing...")
        else:
            print("--- {} folder found. Continuing without wipe...".format(name))
            
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
        if not len(json) or not len(bval) or not len(bvec) or not len(t1) or not len(fmri):
            print("Subject {} does not have all files. Not appending to subjects".format(dwi_name))
            continue

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

# Function to get the atlas choice
def get_atlas_choice(FILTERED_SUBJECT_LIST, ATLAS_FILES):
    # Get the atlas name from the first subject
    atlas_name = get_filename(FILTERED_SUBJECT_LIST[0], "fmri")["fmri"].split("/")[-1].split("_")[1]
    atlas_name = "schaefer100"

    # If the atlas name is not in the atlas files, exit the program
    if not any(atlas_name in atlas_file for atlas_file in ATLAS_FILES):
        print("Atlas name {} not found in atlas files. Please add atlas file".format(atlas_name))
        sys.exit('Exiting program')
    else:
        # Get the atlas path
        atlas_path = [atlas_file for atlas_file in ATLAS_FILES if atlas_name in atlas_file][0]
        print("Atlas path: {}".format(atlas_path))
    
    # Return the atlas path
    return atlas_path