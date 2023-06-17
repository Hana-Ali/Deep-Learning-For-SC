"""
This Python file contains the main pipeline used for tractography
"""

# Import libraries
import subprocess
import glob 
import sys
import os

# --------------- Folder Paths and Data Checking --------------- #

# Check that diffusion data exists - make into loop later
# DWI_INPUT_FOLDER = "/home/hsa22/ConnectomePreprocessing/aamod_get_dicom_diffusion_00001/{}/Diffusion".format(SUBJECT)
# DWI_OUTPUT_FOLDER = "/home/hsa22/ConnectomePreprocessing/aamod_get_dicom_diffusion_00001/{}/SRC".format(SUBJECT)

# Depending on HPC or local machine, change the paths
hpc = False
if hpc == True:
    DWI_INPUT_FOLDER = "/rds/general/user/hsa22/home/dissertation/tractography"
    DWI_OUTPUT_FOLDER = "/rds/general/user/hsa22/home/dissertation/tractography/SRC"

    DSI_COMMAND = "singularity exec dsistudio_latest.sif dsi_studio"

    ATLAS_FOLDER = "/home/hsa22/ConnectomePreprocessing/atlas"
    TRACT_FOLDER = "/home/hsa22/ConnectomePreprocessing/tracts"

else:
    DWI_INPUT_FOLDER = "C:\\Users\\shahi\\OneDrive - Imperial College London\\Documents\\imperial\\Dissertation\\Notebooks\\MyCodes\\tractography"
    DWI_OUTPUT_FOLDER = "C:\\Users\\shahi\\OneDrive - Imperial College London\\Documents\\imperial\\Dissertation\\Notebooks\\MyCodes\\tractography\\dsi_outputs"

    DSI_COMMAND = "dsi_studio"

    ATLAS_FOLDER = "C:\\Users\\shahi\\OneDrive - Imperial College London\\Documents\\imperial\\Dissertation\\Notebooks\\MyCodes\\tractography\\atlas"
    TRACT_FOLDER = "C:\\Users\\shahi\\OneDrive - Imperial College London\\Documents\\imperial\\Dissertation\\Notebooks\\MyCodes\\tractography\\tracts"
    
# Define what atlases to use
atlas_names = ["aal116_mni.nii", "schaefer100_mni.nii.gz"]

# Check if the input folder exists - exit if not
if not os.path.exists(DWI_INPUT_FOLDER):
    print("Diffusion data not found. Please create folder: " + DWI_INPUT_FOLDER)
    sys.exit('Exiting program')
if not os.path.exists(ATLAS_FOLDER):
    print("Atlas folder not found. Please create folder: " + ATLAS_FOLDER)
    sys.exit('Exiting program')
# For the output files - create if not
if not os.path.exists(DWI_OUTPUT_FOLDER):
    print("Output folder not found. Created folder: " + DWI_OUTPUT_FOLDER)
    os.makedirs(DWI_OUTPUT_FOLDER)
if not os.path.exists(TRACT_FOLDER):
    print("Tract folder not found. Created folder: " + TRACT_FOLDER)
    os.makedirs(TRACT_FOLDER)


# -------------------------------------------------- PEDRO COMMANDS -------------------------------------------------- #
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

# --------------- DSI STUDIO Fitting and Tract Reconstruction command --------------- #

# PEDRO_SRC = "dsi_studio --action=src --source=dwi_clean.nii --bval=bvals --bvec=bvecs --output=dwi_clean.src.gz"
TEMP_SRC = "996782"
PEDRO_RECONSTRUCTION_DTI = "{} --action=rec --source={}.src.gz --method=1 --record_odf=1 \
      --param0=1.25 --motion_correction=0 --output=dwi_clean_dti.fib.gz > logs/recon_log.txt".format(DSI_COMMAND, TEMP_SRC)
PEDRO_EXPORT = "{} --action=exp --source=dwi_clean_dti.fib.gz --export=fa > logs/exportingDTI_log.txt".format(DSI_COMMAND)
PEDRO_RECONSTRUCTION_QSDR = "{} --action=rec --source={}.src.gz --method=7 --record_odf=1 \
      --param0=1.25 --motion_correction=0 --other_image=fa:dwi_clean_dti.fib.gz.fa.nii.gz --output=dwi_clean_qsdr.fib.gz \
        > logs/qsdr_log.txt".format(DSI_COMMAND, TEMP_SRC)
# Calling the subprocesses
# subprocess.call(PEDRO_SRC, shell=True)
print("Started reconstruction DTI")
subprocess.call(PEDRO_RECONSTRUCTION_DTI, shell=True)
print("Started exporting DTI")
subprocess.call(PEDRO_EXPORT, shell=True)
print("Started reconstruction QSDR")
subprocess.call(PEDRO_RECONSTRUCTION_QSDR, shell=True)

# --------------- Exporting DSI Metrics command --------------- #
# Export quantities of interest: metrics in NIFTI, connectivity matrix, and tract statistics 
EXPORTING = "{} --action=exp --source=dwi_clean_qsdr.fib.gz --export=qa,rdi,fa,md > logs/exporting_log.txt".format(DSI_COMMAND)
# Calling the subprocess
print("Started exporting metrics")
subprocess.call(EXPORTING, shell=True)

# --------------- Tractography reconstruction command --------------- #
TRACTOGRAPHY = "{} --action=trk --source=dwi_clean_qsdr.fib.gz --fiber_count=1000000 --output=no_file \
    --method=0 --interpolation=0 --max_length=400 --min_length=10 --otsu_threshold=0.6 --random_seed=0 --turning_angle=55 \
        --smoothing=0 --step_size=1 --connectivity=aal116_mni.nii.gz,schaefer100_mni.nii.gz --connectivity_type=end \
            --connectivity_value=count --connectivity_threshold=0.001 > logs/tract.txt".format(DSI_COMMAND)
# Calling the subprocess
print("Started tractography")
subprocess.call(TRACTOGRAPHY, shell=True)

# --------------- Cleaning tractography reconstruction command --------------- #
for tract in glob.glob("{}/*trk.gz".format(TRACT_FOLDER)):
    print("Cleaning tract: {}".format(tract))
    tract_filename = tract.split("/")[-1]
    tract_filename = tract_filename.replace(".trk.gz", "")
    CLEANING = "{} --action=ana --source=dwi_clean_qsdr.fib.gz --tract={} --export=stat".format(DSI_COMMAND, tract_filename)
    # Calling the subprocess
    subprocess.call(CLEANING, shell=True)
    
##########################################################################################################################################

# -------------------------------------------------- DSI STUDIO Tractography commands -------------------------------------------------- #
# Commands from the website:
# https://sites.google.com/a/labsolver.org/dsi-studio/Manual/command-line-for-dsi-studio#TOC-Generate-SRC-files-from-DICOM-NIFTI-2dseq-images


# --------------- Define SRC Generation from DICOM/NIFTI Images command --------------- #
"""
-> action: use "src" for generating src file.
-> source: assign the directory that stores the DICOM files or the file name for 4D nifti file.
        It supports wildcard (e.g., --source=*.nii.gz). 
        If there are more than one 4D NIFTI file, you can combine them by --source=file1.nii.gz,file2.nii.gz
-> output: assign the output src file name.
-> b_table: assign the replacement b-table
-> bval: assign the b value text file
-> bvec: assign the b vector text file
-> recursive: search files in the subdirectories. e.g. "--recursive=1".
-> up_sampling: upsampling the DWI. 
        --up_sampling=0 : no upsampling
        --up_sampling=1 : upsampling by 2
        --up_sampling=2 : downsampling by 2
        --up_sampling=3 : upsampling by 4
        --up_sampling=4 : downsampling by 4
"""


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
"""
-> action: assign "qc" (e.g. --action=qa)
-> source: assign an SRC file or a directory that stores the SRC files 
"""
# QUALITY_CHECK_SRC = "dsi_studio --action=qc --source=folder_storing_sc"

# --------------- Atlas related 