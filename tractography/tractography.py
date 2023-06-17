"""
This Python file contains the main pipeline used for tractography
"""

# Import libraries
import subprocess
import glob 
import sys
import os

# -------------------------------------------------- Folder Paths and Data Checking -------------------------------------------------- #

# --------------- Defining folders and paths based on HPC --------------- #
hpc = False
if hpc == True:
    DWI_MAIN_FOLDER = "/rds/general/user/hsa22/home/dissertation/tractography"
    DWI_OUTPUT_FOLDER = "/rds/general/user/hsa22/home/dissertation/tractography/dsi_outputs"
    DWI_LOGS_FOLDER = "/rds/general/user/hsa22/home/dissertation/tractography/logs"

    DSI_COMMAND = "singularity exec dsistudio_latest.sif dsi_studio"

    ATLAS_FOLDER = "/home/hsa22/ConnectomePreprocessing/atlas"
    TRACT_FOLDER = "/home/hsa22/ConnectomePreprocessing/tracts"

else:
    DWI_MAIN_FOLDER = os.path.realpath(r"C:\\tractography\\subjects")
    DWI_OUTPUT_FOLDER = os.path.realpath(r"C:\\tractography\\dsi_outputs")
    DWI_LOGS_FOLDER = os.path.realpath(r"C:\\tractography\\logs")

    DSI_COMMAND = "dsi_studio"

    ATLAS_FOLDER = os.path.realpath(r"C:\\tractography\\atlas")
    TRACT_FOLDER = os.path.realpath(r"C:\\tractography\\tracts")

# Check if ATLAS folder exists - if not, exit program
if not os.path.exists(ATLAS_FOLDER):
    print("Atlas folder not found. Please create folder: " + ATLAS_FOLDER)
    sys.exit('Exiting program')
else:
    print("Atlas folder found. Continuing...")

# If output folderes don't exist, create them
if not os.path.exists(DWI_OUTPUT_FOLDER):
    print("Output folder not found. Created folder: " + DWI_OUTPUT_FOLDER)
    os.makedirs(DWI_OUTPUT_FOLDER)
else:
    print("Output folder found. Continuing...")
if not os.path.exists(DWI_LOGS_FOLDER):
    print("Logs folder not found. Created folder: " + DWI_LOGS_FOLDER)
    os.makedirs(DWI_LOGS_FOLDER)
else:
    print("Logs folder found. Continuing...")
if not os.path.exists(TRACT_FOLDER):
    print("Tract folder not found. Created folder: " + TRACT_FOLDER)
    os.makedirs(TRACT_FOLDER)
else:
    print("Tract folder found. Continuing...")
    
# --------------- Get DWI, BVAL, BVEC from subdirectories --------------- #
DWI_INPUT_FILES = []
B_VAL_FILES = []
B_VEC_FILES = []
for dwi in glob.glob(os.path.join(DWI_MAIN_FOLDER, os.path.join("**", "*.nii.gz")), recursive=True):
    DWI_INPUT_FILES.append(dwi)
for bval in glob.glob(os.path.join(DWI_MAIN_FOLDER, os.path.join("**", "*.bval")), recursive=True):
    B_VAL_FILES.append(bval)
for bvec in glob.glob(os.path.join(DWI_MAIN_FOLDER, os.path.join("**", "*.bvec")), recursive=True):
    B_VEC_FILES.append(bvec)


# If no files are found - exit the program
if len(DWI_INPUT_FILES) == 0:
    print("No DWI files found. Please add DWI files to the folder: " + DWI_MAIN_FOLDER)
    sys.exit('Exiting program')
else:
    print("DWI files found. Continuing...")
if len(B_VAL_FILES) == 0:
    print("No BVAL files found. Please add BVAL files to the folder: " + DWI_MAIN_FOLDER)
    sys.exit('Exiting program')
else:
    print("BVAL files found. Continuing...")
if len(B_VEC_FILES) == 0:
    print("No BVEC files found. Please add BVEC files to the folder: " + DWI_MAIN_FOLDER)
    sys.exit('Exiting program')
else:
    print("BVEC files found. Continuing...")

# --------------- Define what atlases to use --------------- #
ATLAS_FILES = []
for atlas in glob.glob(os.path.join(ATLAS_FOLDER, "*.nii.gz")):
    ATLAS_FILES.append(atlas)
ATLAS_STRING = ",".join(ATLAS_FILES)

# Check if atlas string is empty
if len(ATLAS_FILES) == 0:
    print("No atlas files found. Please add atlas files to the folder: " + ATLAS_FOLDER)
    sys.exit('Exiting program')
else:
    print("Atlas files found. Continuing...")


# -------------------------------------------------- PEDRO COMMANDS -------------------------------------------------- #

# --------------- DSI STUDIO Fitting and Tract Reconstruction command --------------- #

# For each one of the DWI files, run the following commands
for idx, dwi in enumerate(DWI_INPUT_FILES):
    # --------------- DSI STUDIO filenames --------------- #
    dwi_filename = dwi.split("\\")[-1]
    print('dwi is {}'.format(dwi))
    print('dwi_filename is {}'.format(dwi_filename))
    dwi_filename = dwi_filename.replace(".nii.gz", "")
    # Get the corresponding bval and bvec files
    bval_path = B_VAL_FILES[idx]
    bvec_path = B_VEC_FILES[idx]
    # Define the output file names
    src_filename = os.path.join(DWI_OUTPUT_FOLDER, "{}_clean".format(dwi_filename))
    dti_filename = os.path.join(DWI_OUTPUT_FOLDER, "{}_dti".format(dwi_filename))
    qsdr_filename = os.path.join(DWI_OUTPUT_FOLDER, "{}_qsdr".format(dwi_filename))
    # Define the log file names
    src_log = os.path.join(DWI_LOGS_FOLDER, "src_log_{}.txt".format(dwi_filename))
    dti_log = os.path.join(DWI_LOGS_FOLDER, "dti_log_{}.txt".format(dwi_filename))
    dti_export_log = os.path.join(DWI_LOGS_FOLDER, "exporting_dti_log_{}.txt".format(dwi_filename))
    qsdr_log = os.path.join(DWI_LOGS_FOLDER, "qsdr_log_{}.txt".format(dwi_filename))
    qsdr_export_log = os.path.join(DWI_LOGS_FOLDER, "exporting_qsdr_log_{}.txt".format(dwi_filename))
    tract_log = os.path.join(DWI_LOGS_FOLDER, "tract_log_{}.txt".format(dwi_filename))
    # --------------- DSI STUDIO reconstruction commands --------------- #
    pedro_src = "{} --action=src --source={} --bval={} --bvec={} --output={} > {}".format(DSI_COMMAND,
                                                    dwi, bval_path, bvec_path, src_filename, src_log)
    pedro_reconstruction_dti = "{} --action=rec --source={}.src.gz --method=1 --record_odf=1 \
        --param0=1.25 --motion_correction=0 --output={}.fib.gz > {}".format(DSI_COMMAND, src_filename, dti_filename, dti_log)
    pedro_export_dti = "{} --action=exp --source={}.fib.gz --export=fa > {}".format(DSI_COMMAND, dti_filename, dti_export_log)
    pedro_reconstruction_qsdr = "{} --action=rec --source={}.src.gz --method=7 --record_odf=1 \
        --param0=1.25 --motion_correction=0 --other_image=fa:{}.fib.gz.fa.nii.gz --output={}.fib.gz \
            > {}".format(DSI_COMMAND, src_filename, dti_filename, qsdr_filename, qsdr_log)
    pedro_export_qsdr = "{} --action=exp --source={}.fib.gz --export=qa,rdi,fa,md > {}".format(DSI_COMMAND, qsdr_filename, qsdr_export_log)
    # --------------- DSI STUDIO calling subprocesses command --------------- #
    # Calling the subprocesses  
    print("Started SRC generation - {}. {}".format(idx, dwi_filename))
    subprocess.run(pedro_src, shell=True)
    print("Started reconstruction DTI - {}. {}".format(idx, dwi_filename))
    subprocess.run(pedro_reconstruction_dti, shell=True)
    print("Started exporting metrics DTI - {}. {}".format(idx, dwi_filename))
    subprocess.run(pedro_export_dti, shell=True)
    print("Started reconstruction QSDR - {}. {}".format(idx, dwi_filename))
    subprocess.run(pedro_reconstruction_qsdr, shell=True)
    print("Started exporting metrics QSDR - {}. {}".format(idx, dwi_filename))
    subprocess.run(pedro_export_qsdr, shell=True)

    # --------------- DSI STUDIO Tractography command --------------- #
    # Deterministic tractography
    pedro_tractography = "{} --action=trk --source={}.fib.gz --fiber_count=1000000 --output=no_file \
        --method=0 --interpolation=0 --max_length=400 --min_length=10 --otsu_threshold=0.6 --random_seed=0 --turning_angle=55 \
            --smoothing=0 --step_size=1 --connectivity={} --connectivity_type=end \
                --connectivity_value=count --connectivity_threshold=0.001 > {}".format(DSI_COMMAND, qsdr_filename, ATLAS_STRING, tract_log)
    
#aal116_mni.nii.gz,schaefer100_mni.nii.gz

# PROBABILISTIC TRACTOGRAPHY


# GLOBAL TRACTOGRAPHY

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