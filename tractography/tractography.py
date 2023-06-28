"""
This Python file contains the main pipeline used for tractography
"""

# Import libraries
import multiprocessing as mp
from tract_helpers import *
from sc_functions import *
import regex as re
import subprocess
import os

# -------------------------------------------------- Functions -------------------------------------------------- #

# CHECK HOW TO DO THIS IN PARALLEL - STARMAP OR IMAP
def parallel_process(SUBJECT_FILES, ATLAS, ATLAS_STRING, MAIN_STUDIO_PATH, MAIN_MRTRIX_PATH, MAIN_FSL_PATH, 
                        DSI_COMMAND):

    # Define the stripped paths indices
    STRIPPED_INDEX = 0
    STRIPPED_MASK_INDEX = 1
    STRIPPED_OVERLAY_INDEX = 2

    # Get the filename for this specific process
    dwi_filename = extract_dwi_filename(get_filename(SUBJECT_FILES, "filename")["filename"])
    t1_filename = extract_t1_fmri_filename(get_filename(SUBJECT_FILES, "t1")["t1"])

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
        ATLAS_STRING
    ]
    # Get the studio commands array
    STUDIO_COMMANDS = define_studio_commands(ARGS_STUDIO)

    # # --------------- MRTRIX reconstruction commands --------------- #
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

    # --------------- Calling subprocesses commands --------------- #
    # Stripping T1
    for (bet_cmd, cmd_name) in BET_COMMANDS:
        print("Started {} - {}".format(cmd_name, t1_filename))
        subprocess.run(bet_cmd, shell=True)

    # Preprocessing and cleaning DWI
    for (mrtrix_cmd, cmd_name) in MRTRIX_CLEAN_COMMANDS:
        print("Started {} - {}".format(cmd_name, dwi_filename))
        subprocess.run(mrtrix_cmd, shell=True)

    # Deterministic tractography
    for (dsi_cmd, cmd_name) in STUDIO_COMMANDS:
        print("Started {} - {}".format(cmd_name, dwi_filename))
        subprocess.run(dsi_cmd, shell=True)

    # Probabilistic and global tractography
    for (mrtrix_cmd, cmd_name) in MRTRIX_COMMANDS:
        print("Started {} - {}".format(cmd_name, dwi_filename))
        subprocess.run(mrtrix_cmd, shell=True)
    

    #################################
    # SHOULD ALSO HAVE

    # SET https://github.com/StongeEtienne/set-nf
    # --> Needs surface

    # GESTA - DEEP AE https://github.com/scil-vital/tractolearn
    # --> Not sure how to run, emailed author

    # TRACK-TO-LEARN - DEEP RL https://github.com/scil-vital/TrackToLearn/tree/main
    # --> Needs fODFs, seeding mask (normal mask whole brain mask dwi2mask or bet BET IN T1 GIVES MASK N SKULL STRIP
    # THEN DWI TO MASK THEN COMPUTE IOU OF BET MASK AND B0 MASK THEN KNOW EVERYTHIN MATCH), and WM mask (visually
    # look at wmfod to see if it makes sense as a mask)
    # Agents used for tracking are constrained by their training regime. For example, the agents provided in 
    # `example_models` were trained on a volume with a resolution of 2mm iso voxels and a step size of 0.75mm 
    # using fODFs of order 6, `descoteaux07` basis. When tracking on arbitrary data, the step-size and fODF 
    # order and basis will be adjusted accordingly automatically. **However**, if using fODFs in the `tournier07` 
    # (coming from MRtrix, for example), you will need to set the `--sh_basis` argument accordingly.

    # CNN-TRANSFORMER - ALSO ML
    # --> Don't have code, emailed author
    


# -------------------------------------------------- Folder Paths and Data Checking -------------------------------------------------- #

def main():
    # Get paths, depending on whether we're in HPC or not
    hpc = False
    mapping_inputs = get_dmri_fmri_arguments(hpc)

    # Use the mapping inputs with starmap to run the parallel processes
    with mp.Pool() as pool:
        print("mapping")
        pool.starmap(parallel_process, list(mapping_inputs))

if __name__ == '__main__':
    mp.freeze_support()
    main()