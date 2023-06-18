"""
This Python file contains the main pipeline used for tractography
"""

# Import libraries
import multiprocessing as mp
from tract_helpers import *
import subprocess
import os

# -------------------------------------------------- Functions -------------------------------------------------- #

# CHECK HOW TO DO THIS IN PARALLEL - STARMAP OR IMAP
def parallel_process(DWI_INPUT_FILE, B_VAL_FILE, B_VEC_FILE, ATLAS_STRING, MAIN_STUDIO_PATH, MAIN_MRTRIX_PATH, DWI_LOGS_FOLDER, DSI_COMMAND):

    # Get the filename for this specific process
    dwi_filename = get_dwi_filename(DWI_INPUT_FILE)

    # Ping beginning or process
    print("Started parallel process - {}".format(dwi_filename))
    
    # --------------- DSI STUDIO reconstruction commands --------------- #
    # Define needed arguments array
    ARGS_STUDIO = [
        MAIN_STUDIO_PATH,
        DWI_LOGS_FOLDER,
        DSI_COMMAND,
        DWI_INPUT_FILE,
        B_VAL_FILE,
        B_VEC_FILE,
        ATLAS_STRING
    ]
    # Get the studio commands array
    STUDIO_COMMANDS = define_studio_commands(ARGS_STUDIO)

    # --------------- MRTRIX reconstruction commands --------------- #
    # Define needed arguments array
    ARGS_MRTRIX = [
        DWI_INPUT_FILE,
        B_VEC_FILE,
        B_VAL_FILE,
        MAIN_MRTRIX_PATH
    ]
    # Get the mrtrix commands array
    MRTRIX_COMMANDS = define_mrtrix_commands(ARGS_MRTRIX)

    # --------------- Calling subprocesses commands --------------- #
    for (dsi_cmd, cmd_name) in STUDIO_COMMANDS:
        print("Started {} - {}".format(cmd_name, dwi_filename))
        subprocess.run(dsi_cmd, shell=True)

    for (mrtrix_cmd, cmd_name) in MRTRIX_COMMANDS:
        print("Started {} - {}".format(cmd_name, dwi_filename))
        subprocess.run(mrtrix_cmd, shell=True)


# -------------------------------------------------- Folder Paths and Data Checking -------------------------------------------------- #

def main():
    # --------------- Defining main folders and paths --------------- #
    # Get paths, depending on whether we're in HPC or not
    hpc = False
    (DWI_MAIN_FOLDER, DWI_OUTPUT_FOLDER, DWI_LOGS_FOLDER, DSI_COMMAND, ATLAS_FOLDER, TRACT_FOLDER, 
        MAIN_STUDIO_PATH, MAIN_MRTRIX_PATH) = get_main_paths(hpc)

    # Check if input folders - if not, exit program
    check_input_folders(DWI_MAIN_FOLDER, "DWI")
    check_input_folders(ATLAS_FOLDER, "Atlas")

    # If output folderes don't exist, create them
    check_output_folders_with_subfolders(DWI_OUTPUT_FOLDER, "DWI output")
    check_output_folders(DWI_LOGS_FOLDER, "Logs")
    check_output_folders(TRACT_FOLDER, "Tracts")
    check_output_folders(MAIN_STUDIO_PATH, "Studio")
    check_output_folders(MAIN_MRTRIX_PATH, "MRtrix")
        
    # --------------- Get DWI, BVAL, BVEC from subdirectories --------------- #
    DWI_INPUT_FILES = glob_files(DWI_MAIN_FOLDER, "nii.gz")
    B_VAL_FILES = glob_files(DWI_MAIN_FOLDER, "bval")
    B_VEC_FILES = glob_files(DWI_MAIN_FOLDER, "bvec")
    
    # If no files are found - exit the program
    check_globbed_files(DWI_INPUT_FILES, "DWI")
    check_globbed_files(B_VAL_FILES, "BVAL")
    check_globbed_files(B_VEC_FILES, "BVEC")

    # --------------- Define what atlases to use --------------- #
    ATLAS_FILES = glob_files(ATLAS_FOLDER, "nii.gz")
    # Exit if no atlases are found
    check_globbed_files(ATLAS_FILES, "Atlas")
    # Create atlas string otherwise
    ATLAS_STRING = ",".join(ATLAS_FILES)


    # -------------------------------------------------- PEDRO COMMANDS -------------------------------------------------- #

    # --------------- DSI STUDIO defining inputs for mapping parallel --------------- #
    mapping_inputs = list(zip(DWI_INPUT_FILES, B_VAL_FILES, B_VEC_FILES, [ATLAS_STRING]*len(DWI_INPUT_FILES), [MAIN_STUDIO_PATH]*len(DWI_INPUT_FILES),
                              [MAIN_MRTRIX_PATH]*len(DWI_INPUT_FILES), [DWI_LOGS_FOLDER]*len(DWI_INPUT_FILES), [DSI_COMMAND]*len(DWI_INPUT_FILES)))

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