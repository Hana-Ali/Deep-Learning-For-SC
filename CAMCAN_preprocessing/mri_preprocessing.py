"""
This Python file contains the main pipeline used for tractography
"""

# Import libraries
import multiprocessing as mp
from py_helpers.general_helpers import *
from dwi_helpers.SC_checkpoints import *
from dwi_helpers.SC_commands import *
from dwi_helpers.SC_paths import *
import regex as re
import subprocess
import os

# -------------------------------------------------- Functions -------------------------------------------------- #

# CHECK HOW TO DO THIS IN PARALLEL - STARMAP OR IMAP
def parallel_process(SUBJECT_FILES, ATLAS, MAIN_STUDIO_PATH, MAIN_MRTRIX_PATH, MAIN_FSL_PATH, 
                        DSI_COMMAND):

    # Define the stripped paths indices
    STRIPPED_INDEX = 0
    STRIPPED_MASK_INDEX = 1
    STRIPPED_OVERLAY_INDEX = 2

    print("ATLAS IN PARALLEL: {}".format(ATLAS))

    # Get the filename for this specific process
    dwi_filename = extract_dwi_filename(get_filename(SUBJECT_FILES, "filename")["filename"])
    t1_filename = extract_t1_filename(get_filename(SUBJECT_FILES, "t1")["t1"])

    # Ping beginning or process
    print("Started parallel process - {}".format(dwi_filename))
    
    # --------------- BET/FSL T1 skull stripping command --------------- #
    # Define needed arguments array
    ARGS_BET = [
        SUBJECT_FILES,
        MAIN_FSL_PATH
    ]
    # Get the bet commands array and the stripped paths
    (BET_COMMANDS, STRIPPED_PATHS) = define_fsl_commands(ARGS_BET)

    # --------------- DSI STUDIO preprocessing commands --------------- #
    # Define needed arguments array
    ARGS_MRTRIX_CLEAN = [
        SUBJECT_FILES,
        MAIN_MRTRIX_PATH
    ]
    # Get the mrtrix commands array
    (MRTRIX_CLEAN_COMMANDS, CLEAN_DWI_PATH, CLEAN_BVAL_FILEPATH, 
     CLEAN_BVEC_FILEPATH) = define_mrtrix_clean_commands(ARGS_MRTRIX_CLEAN)
    
    # Define list of clean stuff
    CLEAN_FILES = [CLEAN_DWI_PATH, CLEAN_BVAL_FILEPATH, CLEAN_BVEC_FILEPATH]

    # --------------- DSI STUDIO reconstruction commands --------------- #
    # Define needed arguments array
    ARGS_STUDIO = [
        SUBJECT_FILES,
        CLEAN_FILES,
        MAIN_STUDIO_PATH,
        DSI_COMMAND,
        ATLAS
    ]
    # Get the studio commands array
    STUDIO_COMMANDS = define_studio_commands(ARGS_STUDIO)

    # --------------- MRTRIX reconstruction commands --------------- #
    # Define needed arguments array
    ARGS_MRTRIX = [
        SUBJECT_FILES,
        CLEAN_FILES,
        MAIN_MRTRIX_PATH,
        STRIPPED_PATHS[STRIPPED_INDEX],
        ATLAS
    ]
    # Get the mrtrix commands array
    MRTRIX_COMMANDS = probabilistic_tractography(ARGS_MRTRIX)

    # --------------- Check whether or not we need to do the processing --------------- #
    # Extract some necessary things
    FILES_NEEDED = ["filename"]
    NEEDED_FILE_PATHS = get_filename(SUBJECT_FILES, FILES_NEEDED)

    # FSL Checking
    FSL_PROCESSING = check_missing_fsl(MAIN_FSL_PATH, dwi_filename)

    # MRTRIX Cleaning Checking
    MRTRIX_CLEANING = check_missing_mrtrix_cleaning(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

    # DSI STUDIO Checking
    STUDIO_PROCESSING = check_missing_dsi_studio(NEEDED_FILE_PATHS, MAIN_STUDIO_PATH, dwi_filename)

    # --------------- Calling subprocesses commands --------------- #
    # If we need to do the processing, then we call the subprocesses
    if FSL_PROCESSING:
        # Stripping T1
        for (bet_cmd, cmd_name) in BET_COMMANDS:
            print("Started {} - {}".format(cmd_name, t1_filename))
            subprocess.run(bet_cmd, shell=True)

    if MRTRIX_CLEANING:
        # Preprocessing and cleaning DWI
        for (mrtrix_cmd, cmd_name) in MRTRIX_CLEAN_COMMANDS:
            print("Started {} - {}".format(cmd_name, dwi_filename))
            subprocess.run(mrtrix_cmd, shell=True)

    if STUDIO_PROCESSING:
        # Deterministic tractography
        for (dsi_cmd, cmd_name) in STUDIO_COMMANDS:
            print("Started {} - {}".format(cmd_name, dwi_filename))
            subprocess.run(dsi_cmd, shell=True)

    # Probabilistic and global tractography
    for (mrtrix_cmd, cmd_name) in MRTRIX_COMMANDS:
        print("Started {} - {}".format(cmd_name, dwi_filename))
        subprocess.run(mrtrix_cmd, shell=True)


# -------------------------------------------------- Folder Paths and Data Checking -------------------------------------------------- #

def main():
    # Get paths, depending on whether we're in HPC or not
    hpc =  int(sys.argv[1])
    (ALL_DATA_FOLDER, SUBJECTS_FOLDER, TRACTOGRAPHY_OUTPUT_FOLDER, NIPYPE_OUTPUT_FOLDER, 
        DWI_MAIN_FOLDER, T1_MAIN_FOLDER, FMRI_MAIN_FOLDER, DSI_COMMAND, ATLAS_FOLDER, 
            MAIN_STUDIO_PATH, MAIN_MRTRIX_PATH, MAIN_FSL_PATH) = get_main_paths(hpc)

    # Check if input folders - if not, exit program
    check_input_folders(DWI_MAIN_FOLDER, "DWI")
    check_input_folders(T1_MAIN_FOLDER, "T1")
    check_input_folders(FMRI_MAIN_FOLDER, "fMRI")
    check_input_folders(ATLAS_FOLDER, "Atlas")

    # If output folderes don't exist, create them
    check_output_folders(TRACTOGRAPHY_OUTPUT_FOLDER, "Tractography output", wipe=False)
    check_output_folders(MAIN_STUDIO_PATH, "Studio", wipe=False)
    check_output_folders(MAIN_MRTRIX_PATH, "MRtrix", wipe=False)
    check_output_folders(MAIN_FSL_PATH, "FSL", wipe=False)
        
    # --------------- Get DWI, BVAL, BVEC, T1, fMRI from subdirectories --------------- #
    DWI_INPUT_FILES = glob_files(DWI_MAIN_FOLDER, "nii.gz")
    DWI_JSON_HEADERS = glob_files(DWI_MAIN_FOLDER, "json")
    B_VAL_FILES = glob_files(DWI_MAIN_FOLDER, "bval")
    B_VEC_FILES = glob_files(DWI_MAIN_FOLDER, "bvec")
    T1_INPUT_FILES = glob_files(T1_MAIN_FOLDER, "nii")
    FMRI_INPUT_FILES = glob_files(FMRI_MAIN_FOLDER, "mat")
    # There may be multiple formats for the atlas, so we get both
    ATLAS_FILES_GZ = glob_files(ATLAS_FOLDER, "nii.gz")
    ATLAS_FILES_NII = glob_files(ATLAS_FOLDER, "nii")
    ATLAS_FILES = ATLAS_FILES_GZ + ATLAS_FILES_NII
    
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

    # There are 584 subjects after filtering
    # print('Number of subjects: {}'.format(len(FILTERED_SUBJECT_LIST)))
    
    # --------------- Mapping subject inputs to the HPC job --------------- #
    if hpc:
        # Get the current subject based on the command-line argument
        subject_idx = int(sys.argv[2])
        # Get the index of the subject in the filtered list
        subject = FILTERED_SUBJECT_LIST[subject_idx]
        # Call the parallel process function on this subject
        parallel_process(subject, ATLAS_CHOSEN, MAIN_STUDIO_PATH, MAIN_MRTRIX_PATH, MAIN_FSL_PATH,
                        DSI_COMMAND)
    else:
        # Get the mapping as a list for multiprocessing to work
        mapping_inputs = list(zip(FILTERED_SUBJECT_LIST, [ATLAS_CHOSEN]*len(FILTERED_SUBJECT_LIST), 
                                [MAIN_STUDIO_PATH]*len(FILTERED_SUBJECT_LIST), [MAIN_MRTRIX_PATH]*len(FILTERED_SUBJECT_LIST), 
                                [MAIN_FSL_PATH]*len(FILTERED_SUBJECT_LIST), [DSI_COMMAND]*len(FILTERED_SUBJECT_LIST)))

        # Use the mapping inputs with starmap to run the parallel processes
        with mp.Pool() as pool:
            pool.starmap(parallel_process, list(mapping_inputs))


if __name__ == '__main__':
    mp.freeze_support()
    main()