"""
This Python file contains the main pipeline used for tractography
"""

# Import libraries
import multiprocessing as mp
from tract_helpers import *
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

    # # --------------- DSI STUDIO reconstruction commands --------------- #
    # # Define needed arguments array
    # ARGS_STUDIO = [
    #     SUBJECT_FILES,
    #     CLEAN_FILES,
    #     MAIN_STUDIO_PATH,
    #     DSI_COMMAND,
    #     ATLAS_STRING
    # ]
    # # Get the studio commands array
    # STUDIO_COMMANDS = define_studio_commands(ARGS_STUDIO)

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
    # for (dsi_cmd, cmd_name) in STUDIO_COMMANDS:
    #     print("Started {} - {}".format(cmd_name, dwi_filename))
    #     subprocess.run(dsi_cmd, shell=True)

    # Probabilistic tractography
    for (mrtrix_cmd, cmd_name) in MRTRIX_COMMANDS:
        print("Started {} - {}".format(cmd_name, dwi_filename))
        subprocess.run(mrtrix_cmd, shell=True)
    
    # Global tractography

    #################################
    # SHOULD ALSO HAVE

    # Global tractography
    # --> In MRtrix, or from the optimization paper

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
    # --------------- Defining main folders and paths --------------- #
    # Get paths, depending on whether we're in HPC or not
    hpc = False
    (DWI_MAIN_FOLDER, T1_MAIN_FOLDER, DWI_OUTPUT_FOLDER, DSI_COMMAND, ATLAS_FOLDER, 
     TRACT_FOLDER, MAIN_STUDIO_PATH, MAIN_MRTRIX_PATH, MAIN_FSL_PATH) = get_main_paths(hpc)

    # Check if input folders - if not, exit program
    check_input_folders(DWI_MAIN_FOLDER, "DWI")
    check_input_folders(T1_MAIN_FOLDER, "T1")
    check_input_folders(ATLAS_FOLDER, "Atlas")

    # If output folderes don't exist, create them
    check_output_folders_with_subfolders(DWI_OUTPUT_FOLDER, "DWI output")
    check_output_folders(TRACT_FOLDER, "Tracts")
    check_output_folders(MAIN_STUDIO_PATH, "Studio")
    check_output_folders(MAIN_MRTRIX_PATH, "MRtrix")
    check_output_folders(MAIN_FSL_PATH, "FSL")
        
    # --------------- Get DWI, BVAL, BVEC from subdirectories --------------- #
    DWI_INPUT_FILES = glob_files(DWI_MAIN_FOLDER, "nii.gz")
    DWI_JSON_HEADERS = glob_files(DWI_MAIN_FOLDER, "json")
    B_VAL_FILES = glob_files(DWI_MAIN_FOLDER, "bval")
    B_VEC_FILES = glob_files(DWI_MAIN_FOLDER, "bvec")
    
    # If no files are found - exit the program
    check_globbed_files(DWI_INPUT_FILES, "DWI")
    check_globbed_files(DWI_JSON_HEADERS, "JSON")
    check_globbed_files(B_VAL_FILES, "BVAL")
    check_globbed_files(B_VEC_FILES, "BVEC")

    # --------------- Get T1 from subdirectories --------------- #
    T1_INPUT_FILES = glob_files(T1_MAIN_FOLDER, "nii")
    # Clean up the template files
    T1_INPUT_FILES = list(filter(lambda x: not re.search('Template', x), T1_INPUT_FILES))
    
    # If no files are found - exit the program
    check_globbed_files(T1_INPUT_FILES, "T1")

    # --------------- Create list of DWI and T1 for each subject --------------- #
    SUBJECT_LISTS = create_subject_list(DWI_INPUT_FILES, DWI_JSON_HEADERS, B_VAL_FILES, 
                                        B_VEC_FILES, T1_INPUT_FILES)
    
    # --------------- Define what atlases to use --------------- #
    ATLAS_FILES = glob_files(ATLAS_FOLDER, "nii.gz")
    # Exit if no atlases are found
    check_globbed_files(ATLAS_FILES, "Atlas")
    print('ATLAS_FILES[0]', ATLAS_FILES[0])
    # Create atlas string otherwise
    ATLAS_STRING = ",".join(ATLAS_FILES)


    # -------------------------------------------------- TRACTOGRAPHY COMMANDS -------------------------------------------------- #

    # --------------- DSI STUDIO defining inputs for mapping parallel --------------- #
    mapping_inputs = list(zip(SUBJECT_LISTS, [ATLAS_FILES[0]]*len(DWI_INPUT_FILES), [ATLAS_STRING]*len(DWI_INPUT_FILES), 
                              [MAIN_STUDIO_PATH]*len(DWI_INPUT_FILES), [MAIN_MRTRIX_PATH]*len(DWI_INPUT_FILES), 
                              [MAIN_FSL_PATH]*len(DWI_INPUT_FILES), [DSI_COMMAND]*len(DWI_INPUT_FILES)))

    # Use the mapping inputs with starmap to run the parallel processes
    with mp.Pool() as pool:
        print("mapping")
        pool.starmap(parallel_process, list(mapping_inputs))

if __name__ == '__main__':
    mp.freeze_support()
    main()

# -------------------------------------------------- PEDRO COMMANDS -------------------------------------------------- #

# --------------- DSI STUDIO Fitting and Tract Reconstruction command --------------- #


# # --------------- Cleaning tractography reconstruction command --------------- #
# for tract in glob.glob("{}/*trk.gz".format(TRACT_FOLDER)):
#     print("Cleaning tract: {}".format(tract))
#     tract_filename = tract.split("/")[-1]
#     tract_filename = tract_filename.replace(".trk.gz", "")
#     CLEANING = "{} --action=ana --source=dwi_clean_qsdr.fib.gz --tract={} --export=stat".format(DSI_COMMAND, tract_filename)
#     # Calling the subprocess
#     subprocess.call(CLEANING, shell=True)
    
##########################################################################################################################################

# -------------------------------------------------- DSI STUDIO Tractography commands -------------------------------------------------- #
# Commands from the website:
# https://sites.google.com/a/labsolver.org/dsi-studio/Manual/command-line-for-dsi-studio#TOC-Generate-SRC-files-from-DICOM-NIFTI-2dseq-images


# # --------------- Define SRC Generation from DICOM/NIFTI Images command --------------- #
# """
# -> action: use "src" for generating src file.
# -> source: assign the directory that stores the DICOM files or the file name for 4D nifti file.
#         It supports wildcard (e.g., --source=*.nii.gz). 
#         If there are more than one 4D NIFTI file, you can combine them by --source=file1.nii.gz,file2.nii.gz
# -> output: assign the output src file name.
# -> b_table: assign the replacement b-table
# -> bval: assign the b value text file
# -> bvec: assign the b vector text file
# -> recursive: search files in the subdirectories. e.g. "--recursive=1".
# -> up_sampling: upsampling the DWI. 
#         --up_sampling=0 : no upsampling
#         --up_sampling=1 : upsampling by 2
#         --up_sampling=2 : downsampling by 2
#         --up_sampling=3 : upsampling by 4
#         --up_sampling=4 : downsampling by 4
# """


# Search DICOM files under assigned directory and output results to 1.src
# SEARCH_DICOM = "dsi_studio --action=src --source={} --output={}\.src.gz".format(DWI_INPUT_FOLDER, DWI_OUTPUT_FOLDER)
# Parse assigned 4D NIFTI file and generate SRC files
# PARSE_NIFTI = "dsi_studio --action=src --source=HCA9992517_V1_MR_dMRI_dir98_AP.nii.gz,HCA9992517_V1_MR_dMRI_dir99_AP.nii.gz"
# Parse all 4D NIFTI files in a folder (each of them has a bval and bvec file that shares a similar file name) and generate corresponding SRC files to a new folder 
# PARSE_NIFTI_FOLDER = "dsi_studio --action=src --source=*.nii.gz --output=/src"
# Call the subprocesses
# subprocess.call(SEARCH_DICOM, shell=True)
# subprocess.call(PARSE_NIFTI, shell=True)
# subprocess.call(PARSE_NIFTI_FOLDER, shell=True)

# --------------- Define Quality Check for SRC Files command --------------- #
# """
# -> action: assign "qc" (e.g. --action=qa)
# -> source: assign an SRC file or a directory that stores the SRC files 
# """
# QUALITY_CHECK_SRC = "dsi_studio --action=qc --source=folder_storing_sc"

# --------------- Atlas related 



# --------------- DWI Data Preprocessing (MRtrix3) command --------------- #
# Putting together the DICOMs and denoising using PCA
# MR_CONVERT = "mrconvert {} dwi.mif".format(DWI_INPUT_FOLDER)
# DENOISE = "dwidenoise dwi.mif dwi_denoised.mif"
# # Calling the subprocesses
# subprocess.call(MR_CONVERT, shell=True)
# subprocess.call(DENOISE, shell=True)

# # Use preprocessing information in the DWI headers (by default mrtrix3 only does inhomogeneity correction if 
# # b=0 volumes have phase contrast, which is not the case for these data
# EDDY_CORRECT = "dwifslpreproc dwi_denoised.mif dwi_eddy.mif"
# # Calling the subprocess
# subprocess.call(EDDY_CORRECT, shell=True)

# # Correct for bias and export for DSI
# BIAS_CORRECT = "dwibiascorrect ants dwi_eddy.mif dwi_eddy_unbiased.mif"
# MR_CONVERT_DSI = "mrconvert dwi_eddy_unbiased.mif dwi_clean.nii.gz -export_grad_fsl bvecs bvals"
# # Calling the subprocesses
# subprocess.call(BIAS_CORRECT, shell=True)
# subprocess.call(MR_CONVERT_DSI, shell=True)