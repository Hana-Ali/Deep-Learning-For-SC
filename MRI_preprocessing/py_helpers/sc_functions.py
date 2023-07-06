from .general_helpers import *
import os

# ------------------------------------------------- PATHS ------------------------------------------------- #
# Get the FSL file paths
def get_fsl_paths(NEEDED_FILE_PATHS, MAIN_FSL_PATH):

    # Extract what we need here from the needed file paths
    filename = NEEDED_FILE_PATHS["filename"]

    # Creating folder for each subject in FSL folder
    SUBJECT_FOLDER_NAME = os.path.join(MAIN_FSL_PATH, filename)
    check_output_folders(SUBJECT_FOLDER_NAME, "FSL subject folder", wipe=False)
    
    # Define the path for skull stripped T1
    SKULL_STRIP_PATH = os.path.join(SUBJECT_FOLDER_NAME, "_skull_strip")

    return (SKULL_STRIP_PATH)

# Get the MRtrix cleaning file paths
def get_mrtrix_clean_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]

    # Creating folder for each subject in MRTRIX folder. DON'T WIPE as it'll be used by other functions
    SUBJECT_FOLDER_NAME = os.path.join(MAIN_MRTRIX_PATH, dwi_filename)
    if not os.path.exists(SUBJECT_FOLDER_NAME):
        print("--- MRtrix subject folder not found. Created folder: {}".format(SUBJECT_FOLDER_NAME))
        os.makedirs(SUBJECT_FOLDER_NAME)
    
    # Creating folder for cleaning in MRTRIX folder. WIPE since this is the only function that uses it
    CLEANING_FOLDER_NAME = os.path.join(SUBJECT_FOLDER_NAME, "cleaning")
    check_output_folders(CLEANING_FOLDER_NAME, "MRtrix cleaning folder", wipe=False)

    # DWI nii -> mif filepath
    INPUT_MIF_PATH = os.path.join(CLEANING_FOLDER_NAME, "{}_clean_input".format(dwi_filename))
    # DWI denoising, eddy correction, and bias correction paths
    DWI_DENOISE_PATH = os.path.join(CLEANING_FOLDER_NAME, "{}_denoise".format(dwi_filename))
    DWI_NOISE_PATH = os.path.join(CLEANING_FOLDER_NAME, "{}_noise".format(dwi_filename))
    DWI_EDDY_PATH = os.path.join(CLEANING_FOLDER_NAME, "{}_eddy".format(dwi_filename))
    DWI_BIAS_PATH = os.path.join(CLEANING_FOLDER_NAME, "{}_bias".format(dwi_filename))
    # DWI mif -> nii filepath
    DWI_CONVERT_PATH = os.path.join(CLEANING_FOLDER_NAME, "{}_clean_output".format(dwi_filename))

    # Return the paths
    return (CLEANING_FOLDER_NAME, INPUT_MIF_PATH, DWI_DENOISE_PATH, DWI_NOISE_PATH, DWI_EDDY_PATH, 
            DWI_BIAS_PATH, DWI_CONVERT_PATH)

# Get the DSI_STUDIO file paths
def get_dsi_studio_paths(NEEDED_FILE_PATHS, MAIN_STUDIO_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]

    # Creating folder for each subject in DSI_STUDIO folder. CAN WIPE as this is the only function that uses it
    SUBJECT_FOLDER_NAME = os.path.join(MAIN_STUDIO_PATH, dwi_filename)
    check_output_folders(SUBJECT_FOLDER_NAME, "DSI Studio subject folder", wipe=False)
    # Creating logs folder for each subject. CAN WIPE as this is the only function that uses it
    SUBJECT_LOGS_FOLDER = os.path.join(SUBJECT_FOLDER_NAME, "logs")
    check_output_folders(SUBJECT_LOGS_FOLDER, "DSI Studio logs folder", wipe=False)

    # Create folders and file paths for eachcommand
    STUDIO_SRC_FOLDER = os.path.join(SUBJECT_FOLDER_NAME, "src")
    STUDIO_DTI_FOLDER = os.path.join(SUBJECT_FOLDER_NAME, "dti")
    STUDIO_QSDR_FOLDER = os.path.join(SUBJECT_FOLDER_NAME, "qsdr")
    check_output_folders(STUDIO_SRC_FOLDER, "DSI Studio src folder", wipe=False)
    check_output_folders(STUDIO_DTI_FOLDER, "DSI Studio dti folder", wipe=False)
    check_output_folders(STUDIO_QSDR_FOLDER, "DSI Studio qsdr folder", wipe=False)

    STUDIO_SRC_PATH = os.path.join(STUDIO_SRC_FOLDER, "{}_clean".format(dwi_filename))
    STUDIO_DTI_PATH = os.path.join(STUDIO_DTI_FOLDER, "{}_dti".format(dwi_filename))
    STUDIO_QSDR_PATH = os.path.join(STUDIO_QSDR_FOLDER, "{}_qsdr".format(dwi_filename))

    # Log folders and file paths
    SRC_LOG_FOLDER = os.path.join(SUBJECT_LOGS_FOLDER, "dwi_to_src")
    DTI_LOG_FOLDER = os.path.join(SUBJECT_LOGS_FOLDER, "src_to_dti")
    QSDR_LOG_FOLDER = os.path.join(SUBJECT_LOGS_FOLDER, "src_to_qsdr")
    TRACT_LOG_FOLDER = os.path.join(SUBJECT_LOGS_FOLDER, "recon_to_tract")
    DTI_EXP_LOG_FOLDER = os.path.join(SUBJECT_LOGS_FOLDER, "dti_export")
    QSDR_EXP_LOG_FOLDER = os.path.join(SUBJECT_LOGS_FOLDER, "qsdr_export")
    check_output_folders(SRC_LOG_FOLDER, "DSI Studio dwi to src logs folder", wipe=False)
    check_output_folders(DTI_LOG_FOLDER, "DSI Studio src to dti logs folder", wipe=False)
    check_output_folders(QSDR_LOG_FOLDER, "DSI Studio src to qsdr logs folder", wipe=False)
    check_output_folders(TRACT_LOG_FOLDER, "DSI Studio recon to tract logs folder", wipe=False)
    check_output_folders(DTI_EXP_LOG_FOLDER, "DSI Studio dti export logs folder", wipe=False)
    check_output_folders(QSDR_EXP_LOG_FOLDER, "DSI Studio qsdr export logs folder", wipe=False)

    SRC_LOG_PATH = os.path.join(SRC_LOG_FOLDER, "src_log_{}.txt".format(dwi_filename))
    DTI_LOG_PATH = os.path.join(DTI_LOG_FOLDER, "dti_log_{}.txt".format(dwi_filename))
    DTI_EXP_LOG_PATH = os.path.join(DTI_EXP_LOG_FOLDER, "exporting_dti_log_{}.txt".format(dwi_filename))
    QSDR_LOG_PATH = os.path.join(QSDR_LOG_FOLDER, "qsdr_log_{}.txt".format(dwi_filename))
    QSDR_EXP_LOG_PATH = os.path.join(QSDR_EXP_LOG_FOLDER, "exporting_qsdr_log_{}.txt".format(dwi_filename))
    TRACT_LOG_PATH = os.path.join(TRACT_LOG_FOLDER, "tract_log_{}.txt".format(dwi_filename))

    # Return the paths
    return (STUDIO_SRC_PATH, STUDIO_DTI_PATH, STUDIO_QSDR_PATH, SRC_LOG_PATH, DTI_LOG_PATH, 
            DTI_EXP_LOG_PATH, QSDR_LOG_PATH, QSDR_EXP_LOG_PATH, TRACT_LOG_PATH)

# Define MRTRIX folder paths
def main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]

    # Creating folder for each subject in MRTRIX folder. DON'T WIPE as it'll be used by other functions
    SUBJECT_FOLDER_NAME = os.path.join(MAIN_MRTRIX_PATH, dwi_filename)
    check_output_folders(SUBJECT_FOLDER_NAME, "MRtrix subject folder", wipe=False)
    
    # Creating folder for cleaning in MRTRIX folder. DON'T WIPE as it'll be used by other functions
    RECON_FOLDER_NAME = os.path.join(SUBJECT_FOLDER_NAME, "reconstruction")
    check_output_folders(RECON_FOLDER_NAME, "MRtrix reconstruction folder", wipe=False)

    # Because there's a lot of commands, we define extra folder paths here. DON'T WIPE as it'll be used by other functions
    RESPONSE_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "response")
    FOD_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "fod")
    FOD_NORM_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "fod_norm")
    T1_REG_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "t1_and_Fivett_reg")
    ATLAS_REG_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "atlas_reg")
    PROB_TRACKING_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "prob_tracking")
    GLOBAL_TRACKING_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "global_tracking")
    CONNECTIVITY_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "connectivity")
    check_output_folders(RESPONSE_FOLDER_NAME, "MRtrix response folder", wipe=False)
    check_output_folders(FOD_FOLDER_NAME, "MRtrix fod folder", wipe=False)
    check_output_folders(FOD_NORM_FOLDER_NAME, "MRtrix fod norm folder", wipe=False)
    check_output_folders(T1_REG_FOLDER_NAME, "MRtrix t1 and Fivett reg folder", wipe=False)
    check_output_folders(ATLAS_REG_FOLDER_NAME, "MRtrix atlas reg folder", wipe=False)
    check_output_folders(PROB_TRACKING_FOLDER_NAME, "MRtrix probabilistic tracking folder", wipe=False)
    check_output_folders(GLOBAL_TRACKING_FOLDER_NAME, "MRtrix global tracking folder", wipe=False)
    check_output_folders(CONNECTIVITY_FOLDER_NAME, "MRtrix connectivity folder", wipe=False)

    # Return the paths
    return (SUBJECT_FOLDER_NAME, RECON_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, FOD_NORM_FOLDER_NAME,
                T1_REG_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, GLOBAL_TRACKING_FOLDER_NAME,
                    CONNECTIVITY_FOLDER_NAME)

# Define MRtrix general paths
def get_mrtrix_general_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]
    
    # Get the folder names
    (_, RECON_FOLDER_NAME, _, _, _, _, _, _, _, _) = main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

    # DWI nii -> mif filepath
    INPUT_MIF_PATH = os.path.join(RECON_FOLDER_NAME, "{}_recon_input".format(dwi_filename))
    # Path of the brain mask
    MASK_MIF_PATH = os.path.join(RECON_FOLDER_NAME, "{}_mask".format(dwi_filename))

    # Return the paths
    return (INPUT_MIF_PATH, MASK_MIF_PATH)
    
# Define MRTrix FOD paths
def get_mrtrix_fod_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]
    
    # Get the folder names
    (_, _, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, FOD_NORM_FOLDER_NAME, _, 
        _, _, _, _) = main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

    # Finding the response function paths
    RESPONSE_WM_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_wm".format(dwi_filename))
    RESPONSE_GM_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_gm".format(dwi_filename))
    RESPONSE_CSF_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_csf".format(dwi_filename))
    RESPONSE_VOXEL_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_response_voxels".format(dwi_filename))
    # Finding the fiber orientation distributions (fODs)
    WM_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_wmfod".format(dwi_filename))
    GM_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_gmfod".format(dwi_filename))
    CSF_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_csffod".format(dwi_filename))
    VF_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_vf".format(dwi_filename))
    # Normalizing the fiber orientation distributions (fODs)
    WM_FOD_NORM_PATH = os.path.join(FOD_NORM_FOLDER_NAME, "{}_wmfod_norm".format(dwi_filename))
    GM_FOD_NORM_PATH = os.path.join(FOD_NORM_FOLDER_NAME, "{}_gmfod_norm".format(dwi_filename))
    CSF_FOD_NORM_PATH = os.path.join(FOD_NORM_FOLDER_NAME, "{}_csffod_norm".format(dwi_filename))
    
    # Return the paths
    return (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH,
                WM_FOD_PATH, GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, CSF_FOD_NORM_PATH)

# Define MRtrix Registration paths
def get_mrtrix_registration_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, ATLAS):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]
    
    # Get the folder names
    (_, _, _, _, _, T1_REG_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, _, _, 
        _) = main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

    # Convert T1 nii to mif, then create 5ttgen, register/map it to the B0 space of DWI, then convert back to nii
    T1_MIF_PATH = os.path.join(T1_REG_FOLDER_NAME, "{}_t1".format(dwi_filename))
    FIVETT_NOREG_PATH = os.path.join(T1_REG_FOLDER_NAME, "{}_5ttgen".format(dwi_filename))
    DWI_B0_PATH = os.path.join(T1_REG_FOLDER_NAME, "{}_b0".format(dwi_filename))
    DWI_B0_NII = os.path.join(T1_REG_FOLDER_NAME, "{}_b0.nii.gz".format(dwi_filename))
    FIVETT_GEN_NII = os.path.join(T1_REG_FOLDER_NAME, "{}_5ttgen.nii.gz".format(dwi_filename))
    T1_DWI_MAP_MAT = os.path.join(T1_REG_FOLDER_NAME, "{}_t12dwi_fsl".format(dwi_filename))
    T1_DWI_CONVERT_INV = os.path.join(T1_REG_FOLDER_NAME, "{}_t12dwi_mrtrix".format(dwi_filename))
    FIVETT_REG_PATH = os.path.join(T1_REG_FOLDER_NAME, "{}_5ttgenreg".format(dwi_filename))
    # Convert atlas nii to mif, then register/map it to the B0 space of DWI
    ATLAS_DWI_MAP_MAT = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlas2dwi_fsl".format(dwi_filename))
    ATLAS_DWI_CONVERT_INV = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlas2dwi_mrtrix".format(dwi_filename))
    ATLAS_REG_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlasreg".format(dwi_filename))
    # Getting the name of the atlas without .nii.gz
    ATLAS_NAME = ATLAS.split("/")[-1].split(".")[0].split("_")[0]
    ATLAS_MIF_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlas_mif".format(ATLAS_NAME))

    # Return the paths
    return (T1_MIF_PATH, FIVETT_NOREG_PATH, DWI_B0_PATH, DWI_B0_NII, FIVETT_GEN_NII, T1_DWI_MAP_MAT,
                T1_DWI_CONVERT_INV, FIVETT_REG_PATH, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, ATLAS_REG_PATH, ATLAS_MIF_PATH)

# Define MRtrix probabilistic tracking paths
def get_mrtrix_probtrack_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]
    
    # Get the folder names
    (_, _, _, _, _, _, _, PROB_TRACKING_FOLDER_NAME, _, _) = main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

    # Mask for streamline seeding paths and probabilistic tractography path
    GM_WM_SEED_PATH = os.path.join(PROB_TRACKING_FOLDER_NAME, "{}_gmwmseed".format(dwi_filename))
    TRACT_TCK_PATH = os.path.join(PROB_TRACKING_FOLDER_NAME, "{}_tract".format(dwi_filename))

    # Return the paths
    return (GM_WM_SEED_PATH, TRACT_TCK_PATH)

# Define MRtrix global tracking paths
def get_mrtrix_global_tracking_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):
    
    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]
    
    # Get the folder names
    (_, _, _, _, _, _, _, _, GLOBAL_TRACKING_FOLDER_NAME, _) = main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

    # FOD and FISO for global tractography paths
    GLOBAL_FOD_PATH = os.path.join(GLOBAL_TRACKING_FOLDER_NAME, "{}_global_fod".format(dwi_filename))
    GLOBAL_FISO_PATH = os.path.join(GLOBAL_TRACKING_FOLDER_NAME, "{}_global_fiso".format(dwi_filename))
    # Global tractography path
    GLOBAL_TRACT_PATH = os.path.join(GLOBAL_TRACKING_FOLDER_NAME, "{}_global_tract".format(dwi_filename))

    # Return the paths
    return (GLOBAL_FOD_PATH, GLOBAL_FISO_PATH, GLOBAL_TRACT_PATH)

# Define MRtrix connectome paths
def get_mrtrix_connectome_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]

    # Get the folder names
    (_, _, _, _, _, _, _, _, _, CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

    # Connectivity matrix path
    CONNECTIVITY_PROB_PATH = os.path.join(CONNECTIVITY_FOLDER_NAME, "{}_prob_connectivity".format(dwi_filename))
    CONNECTIVITY_GLOBAL_PATH = os.path.join(CONNECTIVITY_FOLDER_NAME, "{}_global_connectivity".format(dwi_filename))

    # Return the paths
    return (CONNECTIVITY_PROB_PATH, CONNECTIVITY_GLOBAL_PATH)

# ------------------------------------------------- COMMANDS ------------------------------------------------- #
# Define FSL commands
def define_fsl_commands(ARGS):
    # Extract arguments needed to define paths
    SUBJECT_FILES = ARGS[0]
    MAIN_FSL_PATH = ARGS[1]

    # Define what's needed for FSL and extract them from subject files
    NEEDED_FILES = ["filename", "t1"]
    NEEDED_FILE_PATHS = get_filename(SUBJECT_FILES, NEEDED_FILES)
    
    # Extract what we need here from the needed file paths
    T1_FILEPATH = NEEDED_FILE_PATHS["t1"]

    # Get the FSL file paths
    (SKULL_STRIP_PATH) = get_fsl_paths(NEEDED_FILE_PATHS, MAIN_FSL_PATH)

    # Define the commands
    SKULL_STRIP_CMD = "bet {input} {output} -f 0.5 -m -o -R -B".format(input=T1_FILEPATH, output=SKULL_STRIP_PATH)

    # Create commands array
    FSL_COMMANDS = [(SKULL_STRIP_CMD, "Skull stripping")]

    # Define the paths to the outputs
    SKULL_STRIP_T1 = SKULL_STRIP_PATH + ".nii.gz"
    SKULL_STRIP_MASK = SKULL_STRIP_PATH + "_mask.nii.gz"
    SKULL_STRIP_OVERLAY = SKULL_STRIP_PATH + "_overlay.nii.gz"

    # Create the paths array
    STRIPPED_T1_PATHS = [SKULL_STRIP_T1, SKULL_STRIP_MASK, SKULL_STRIP_OVERLAY]

    # Return the commands array
    return (FSL_COMMANDS, STRIPPED_T1_PATHS)

# Define MRtrix cleaning commands
def define_mrtrix_clean_commands(ARGS):
    # Extract arguments needed to define paths
    SUBJECT_FILES = ARGS[0]
    MAIN_MRTRIX_PATH = ARGS[1]

    # Define what's needed for MRTrix cleaning and extract them from subject files
    CLEANING_NEEDED = ["filename", "dwi", "json", "bval", "bvec"]
    NEEDED_FILE_PATHS = get_filename(SUBJECT_FILES, CLEANING_NEEDED)

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS['filename']
    dwi = NEEDED_FILE_PATHS['dwi']
    json = NEEDED_FILE_PATHS['json']
    bval = NEEDED_FILE_PATHS['bval']
    bvec = NEEDED_FILE_PATHS['bvec']

    # Get the rest of the paths for the commands
    (CLEANING_FOLDER_NAME, INPUT_MIF_PATH, DWI_DENOISE_PATH, DWI_NOISE_PATH, DWI_EDDY_PATH, 
     DWI_BIAS_PATH, DWI_CONVERT_PATH) = get_mrtrix_clean_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    
    # Define some extra paths to the outputs
    CLEAN_BVAL_FILEPATH = os.path.join(CLEANING_FOLDER_NAME, "clean_bval")
    CLEAN_BVEC_FILEPATH = os.path.join(CLEANING_FOLDER_NAME, "clean_bvec")

    # Define the commands
    MIF_CMD = "mrconvert {input_nii} -fslgrad {bvec} {bval} -json_import {json} {output}.mif".format(
        input_nii=dwi, bvec=bvec, bval=bval, json=json, output=INPUT_MIF_PATH)
    DWI_DENOISE_CMD = "dwidenoise {input}.mif {output}.mif -noise {noise}.mif".format(
        input=INPUT_MIF_PATH, output=DWI_DENOISE_PATH, noise=DWI_NOISE_PATH)
    DWI_EDDY_CMD = 'dwifslpreproc {input}.mif {output}.mif -fslgrad {bvec} {bval} -eddy_options " --slm=linear" -rpe_header'.format(
        input=DWI_DENOISE_PATH, bvec=bvec, bval=bval, output=DWI_EDDY_PATH)
    DWI_BIAS_CMD = "dwibiascorrect ants {input}.mif {output}.mif".format(input=DWI_EDDY_PATH, output=DWI_BIAS_PATH)
    DWI_CONVERT_CMD = "mrconvert {input}.mif {output}.nii.gz -export_grad_fsl {bvec_clean}.bvec {bval_clean}.bval".format(
        input=DWI_BIAS_PATH, output=DWI_CONVERT_PATH, bvec_clean=CLEAN_BVEC_FILEPATH, bval_clean=CLEAN_BVAL_FILEPATH)

    # Create commands array
    MRTRIX_COMMANDS = [(MIF_CMD, "Conversion NifTI -> MIF"), (DWI_DENOISE_CMD, "Denoising DWI"),
                          (DWI_EDDY_CMD, "Eddy correction"), (DWI_BIAS_CMD, "Bias correction"),
                            (DWI_CONVERT_CMD, "Conversion MIF -> NifTI")]

    # Define the paths to the outputs
    CLEAN_DWI_PATH = DWI_CONVERT_PATH + ".nii.gz"
    CLEAN_BVAL_FILEPATH = CLEAN_BVAL_FILEPATH + ".bval"
    CLEAN_BVEC_FILEPATH = CLEAN_BVEC_FILEPATH + ".bvec"

    # Return the commands array
    return (MRTRIX_COMMANDS, CLEAN_DWI_PATH, CLEAN_BVAL_FILEPATH, CLEAN_BVEC_FILEPATH)

# Define DSI_STUDIO commands
def define_studio_commands(ARGS):
    # Extract arguments needed to define commands
    SUBJECT_FILES = ARGS[0]
    CLEAN_FILES = ARGS[1]
    MAIN_STUDIO_PATH = ARGS[2]
    DSI_COMMAND = ARGS[3]
    ATLAS_CHOSEN = ARGS[4]
    
    # Define what's needed for DSI STUDIO and extract them from subject files
    CLEANING_NEEDED = ["filename", "dwi", "bval", "bvec"]
    NEEDED_FILE_PATHS = get_filename(SUBJECT_FILES, CLEANING_NEEDED)

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]
    dwi = NEEDED_FILE_PATHS["dwi"]
    bval = NEEDED_FILE_PATHS["bval"]
    bvec = NEEDED_FILE_PATHS["bvec"]

    # Extract the clean files
    CLEAN_DWI_PATH = CLEAN_FILES[0]
    CLEAN_BVAL_FILEPATH = CLEAN_FILES[1]
    CLEAN_BVEC_FILEPATH = CLEAN_FILES[2]

    # Get the rest of the paths for the commands
    (SRC_PATH, DTI_PATH, QSDR_PATH, SRC_LOG_PATH, DTI_LOG_PATH, DTI_EXP_LOG_PATH,
            QSDR_LOG_PATH, QSDR_EXP_LOG_PATH, TRACT_LOG_PATH) = get_dsi_studio_paths(NEEDED_FILE_PATHS, MAIN_STUDIO_PATH)
    
    # Define the commands
    SRC_CMD = "{command} --action=src --source={dwi} --bval={bval} --bvec={bvec} --output={output} > {log}".format(
        command=DSI_COMMAND, dwi=CLEAN_DWI_PATH, bval=CLEAN_BVAL_FILEPATH, bvec=CLEAN_BVEC_FILEPATH, 
        output=SRC_PATH, log=SRC_LOG_PATH)
    DTI_CMD = "{command} --action=rec --source={src}.src.gz --method=1 --record_odf=1 \
        --param0=1.25 --motion_correction=0 --output={output}.fib.gz > {log}".format(
        command=DSI_COMMAND, src=SRC_PATH, output=DTI_PATH, log=DTI_LOG_PATH)
    EXPORT_DTI_CMD = "{command} --action=exp --source={dti}.fib.gz --export=fa > {log}".format(
        command=DSI_COMMAND, dti=DTI_PATH, log=DTI_EXP_LOG_PATH)
    QSDR_CMD = "{command} --action=rec --source={src}.src.gz --method=7 --record_odf=1 \
        --param0=1.25 --motion_correction=0 --other_image=fa:{dti}.fib.gz.fa.nii.gz --output={output}.fib.gz \
            > {log}".format(command=DSI_COMMAND, src=SRC_PATH, dti=DTI_PATH, output=QSDR_PATH, log=QSDR_LOG_PATH)
    EXPORT_QSDR_CMD = "{command} --action=exp --source={qsdr}.fib.gz --export=qa,rdi,fa,md > {log}".format(
        command=DSI_COMMAND, qsdr=QSDR_PATH, log=QSDR_EXP_LOG_PATH)

    STUDIO_DET_TRACT_CMD = "{command} --action=trk --source={qsdr}.fib.gz --fiber_count=1000000 --output=no_file \
    --method=0 --interpolation=0 --max_length=400 --min_length=10 --otsu_threshold=0.6 --random_seed=0 --turning_angle=55 \
        --smoothing=0 --step_size=1 --connectivity={atlas_chosen} --connectivity_type=end \
            --connectivity_value=count --connectivity_threshold=0.001 > {log}".format(
        command=DSI_COMMAND, qsdr=QSDR_PATH, atlas_chosen=ATLAS_CHOSEN, log=TRACT_LOG_PATH)

    # Create commands array
    STUDIO_COMMANDS = [(SRC_CMD, "SRC creation"), (DTI_CMD, "DTI evaluation"), 
                       (EXPORT_DTI_CMD, "Exporting DTI metrics"), (QSDR_CMD, "QSDR evaluation"), 
                       (EXPORT_QSDR_CMD, "Exporting QSDR metrics"), (STUDIO_DET_TRACT_CMD, "Deterministic tractography")]

    # Return the commands array
    return STUDIO_COMMANDS

# Define MRtrix FOD commands
def define_mrtrix_fod_commands(ARGS):

    # Extract arguments needed to define paths
    SUBJECT_FILES = ARGS[0]
    CLEAN_FILES = ARGS[1]
    MAIN_MRTRIX_PATH = ARGS[2]

    # Define what's needed for DSI STUDIO and extract them from subject files
    FILES_NEEDED = ["filename", "dwi", "bval", "bvec"]
    NEEDED_FILE_PATHS = get_filename(SUBJECT_FILES, FILES_NEEDED)

    # Extract the clean files
    CLEAN_DWI_PATH = CLEAN_FILES[0]
    CLEAN_BVAL_FILEPATH = CLEAN_FILES[1]
    CLEAN_BVEC_FILEPATH = CLEAN_FILES[2]

    # Get the rest of the paths for the commands
    (INPUT_MIF_PATH, MASK_MIF_PATH) = get_mrtrix_general_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Get the fod paths
    (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH,
        WM_FOD_PATH, GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, 
            CSF_FOD_NORM_PATH) = get_mrtrix_fod_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)


    # DWI nii -> mif command
    MIF_CMD = "mrconvert {input_nii} -fslgrad {bvec} {bval} {output}.mif".format(input_nii=CLEAN_DWI_PATH, 
                                                                                bvec=CLEAN_BVEC_FILEPATH, 
                                                                                bval=CLEAN_BVAL_FILEPATH, 
                                                                                output=INPUT_MIF_PATH)
    # DWI brain mask command
    MASK_CMD = "dwi2mask {input}.mif {output}.mif".format(input=INPUT_MIF_PATH, output=MASK_MIF_PATH)
    # Response estimation of WM, GM, CSF from DWI command
    RESPONSE_EST_CMD = "dwi2response dhollander {input}.mif -mask {mask}.mif {wm}.txt {gm}.txt {csf}.txt -voxels {response_voxels}.mif".format(
        input=INPUT_MIF_PATH, mask=MASK_MIF_PATH, wm=RESPONSE_WM_PATH, gm=RESPONSE_GM_PATH, csf=RESPONSE_CSF_PATH, response_voxels=RESPONSE_VOXEL_PATH)
    VIEW_RESPONSE_CMD = "mrview {input}.mif -overlay.load {response_voxels}.mif".format(input=INPUT_MIF_PATH,
                                                                                        response_voxels=RESPONSE_VOXEL_PATH)    
    # Spherical deconvolution to estimate fODs command
    MULTISHELL_CSD_CMD = "dwi2fod msmt_csd {input}.mif {wm}.txt {wmfod}.mif {gm}.txt {gmfod}.mif {csf}.txt \
        {csffod}.mif -mask {mask}.mif".format(
        input=INPUT_MIF_PATH, wm=RESPONSE_WM_PATH, wmfod=WM_FOD_PATH, gm=RESPONSE_GM_PATH, gmfod=GM_FOD_PATH,
        csf=RESPONSE_CSF_PATH, csffod=CSF_FOD_PATH, mask=MASK_MIF_PATH)
    # Combining fODs into a VF command
    COMBINE_FODS_CMD = "mrconvert -coord 3 0 {wmfod}.mif - | mrcat {csffod}.mif {gmfod}.mif - {output}.mif".format(
        wmfod=WM_FOD_PATH, csffod=CSF_FOD_PATH, gmfod=GM_FOD_PATH, output=VF_FOD_PATH)
    VIEW_COMBINED_FODS_CMD = "mrview {vf}.mif -odf.load_sh {wmfod}.mif".format(vf=VF_FOD_PATH, wmfod=WM_FOD_PATH)
    # Normalizing fODs command
    NORMALIZE_FODS_CMD = "mtnormalise {wmfod}.mif {wmfod_norm}.mif {gmfod}.mif {gmfod_norm}.mif {csffod}.mif \
        {csffod_norm}.mif -mask {mask}.mif".format(
        wmfod=WM_FOD_PATH, wmfod_norm=WM_FOD_NORM_PATH, gmfod=GM_FOD_PATH, gmfod_norm=GM_FOD_NORM_PATH, csffod=CSF_FOD_PATH,
        csffod_norm=CSF_FOD_NORM_PATH, mask=MASK_MIF_PATH)
    VIEW_NORMALIZED_FODS_CMD = "mrview {vf}.mif -odf.load_sh {wmfod_norm}.mif".format(vf=VF_FOD_PATH,
                                                                                      wmfod_norm=WM_FOD_NORM_PATH)
    
    # Return the commands
    return (MIF_CMD, MASK_CMD, RESPONSE_EST_CMD, VIEW_RESPONSE_CMD, MULTISHELL_CSD_CMD, COMBINE_FODS_CMD,
                VIEW_COMBINED_FODS_CMD, NORMALIZE_FODS_CMD, VIEW_NORMALIZED_FODS_CMD)

# Define MRtrix Registration commands
def define_mrtrix_registration_commands(ARGS):

    # Extract arguments needed to define paths
    SUBJECT_FILES = ARGS[0]
    MAIN_MRTRIX_PATH = ARGS[1]
    STRIPPED_T1_PATH = ARGS[2]
    ATLAS = ARGS[3]

    # Define what's needed for DSI STUDIO and extract them from subject files
    FILES_NEEDED = ["filename", "dwi", "bval", "bvec"]
    NEEDED_FILE_PATHS = get_filename(SUBJECT_FILES, FILES_NEEDED)

    # Get the rest of the paths for the commands
    (INPUT_MIF_PATH, _) = get_mrtrix_general_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Get the registration paths
    (T1_MIF_PATH, FIVETT_NOREG_PATH, DWI_B0_PATH, DWI_B0_NII, FIVETT_GEN_NII, T1_DWI_MAP_MAT,
        T1_DWI_CONVERT_INV, FIVETT_REG_PATH, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, ATLAS_REG_PATH, 
            ATLAS_MIF_PATH) = get_mrtrix_registration_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, ATLAS)

    # T1 nii -> mif command
    MIF_T1_CMD = "mrconvert {input_nii} {output}.mif".format(input_nii=STRIPPED_T1_PATH, output=T1_MIF_PATH)
    # 5ttgen creation with no registration command
    FIVETT_NOREG_CMD = "5ttgen fsl {input}.mif {output}.mif".format(input=T1_MIF_PATH, output=FIVETT_NOREG_PATH)
    # Extracting mean B0 and transforming to NII command
    DWI_B0_CMD = "dwiextract {input}.mif - -bzero | mrmath - mean {output}.mif -axis 3".format(
        input=INPUT_MIF_PATH, output=DWI_B0_PATH)
    DWI_B0_NII_CMD = "mrconvert {input}.mif {output}".format(input=DWI_B0_PATH, output=DWI_B0_NII)
    FIVETT_GEN_NII_CMD = "mrconvert {input}.mif {output}".format(input=FIVETT_NOREG_PATH, output=FIVETT_GEN_NII)
    # Transformation and registration of T1 to DWI space commands
    REGISTER_T1_DWI_CMD = "flirt -in {dwi} -ref {fivett} -interp nearestneighbour -dof 6 -omat {transform_mat}.mat".format(
        dwi=DWI_B0_NII, fivett=FIVETT_GEN_NII, transform_mat=T1_DWI_MAP_MAT)
    TRANSFORMATION_T1_DWI_CMD = "transformconvert {transform_mat}.mat {dwi} {fivett} flirt_import {output}.txt".format(
        transform_mat=T1_DWI_MAP_MAT, dwi=DWI_B0_NII, fivett=FIVETT_GEN_NII, output=T1_DWI_CONVERT_INV)
    FINAL_TRANSFORM_CMD = "mrtransform {fivett}.mif -linear {transform}.txt -inverse {output}.mif".format(
        fivett=FIVETT_NOREG_PATH, transform=T1_DWI_CONVERT_INV, output=FIVETT_REG_PATH)
    # Transformation and registration of atlas to DWI space (to be used for connectome generation)
    REGISTER_ATLAS_DWI_CMD = "flirt -in {dwi} -ref {atlas} -interp nearestneighbour -dof 6 -omat {transform_mat}.mat".format(
        dwi=DWI_B0_NII, atlas=ATLAS, transform_mat=ATLAS_DWI_MAP_MAT)
    TRANSFORMATION_ATLAS_DWI_CMD = "transformconvert {transform_mat}.mat {dwi} {atlas} flirt_import {output}.txt".format(
        transform_mat=ATLAS_DWI_MAP_MAT, dwi=DWI_B0_NII, atlas=ATLAS, output=ATLAS_DWI_CONVERT_INV)
    ATLAS_MIF_CMD = "mrconvert {atlas} {output}.mif".format(atlas=ATLAS, output=ATLAS_MIF_PATH)
    FINAL_ATLAS_TRANSFORM_CMD = "mrtransform {atlas}.mif -linear {transform}.txt -inverse {output}.mif".format(
        atlas=ATLAS_MIF_PATH, transform=ATLAS_DWI_CONVERT_INV, output=ATLAS_REG_PATH)
    
    # Return the commands
    return (MIF_T1_CMD, FIVETT_NOREG_CMD, DWI_B0_CMD, DWI_B0_NII_CMD, FIVETT_GEN_NII_CMD, REGISTER_T1_DWI_CMD,
                TRANSFORMATION_T1_DWI_CMD, FINAL_TRANSFORM_CMD, REGISTER_ATLAS_DWI_CMD, TRANSFORMATION_ATLAS_DWI_CMD,
                    ATLAS_MIF_CMD, FINAL_ATLAS_TRANSFORM_CMD)

# Define MRtrix probabilistic tracking commands
def define_mrtrix_probtrack_commands(ARGS):
    
    # Extract arguments needed to define paths
    SUBJECT_FILES = ARGS[0]
    MAIN_MRTRIX_PATH = ARGS[1]
    ATLAS = ARGS[2]

    # Define what's needed for to extract from subject files
    FILES_NEEDED = ["filename", "dwi", "bval", "bvec"]
    NEEDED_FILE_PATHS = get_filename(SUBJECT_FILES, FILES_NEEDED)

    # Get the registration paths
    (_, _, _, _, _, _, _, FIVETT_REG_PATH, _, _, _, 
        _) = get_mrtrix_registration_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, ATLAS)
    # Get the fod paths
    (_, _, _, _, _, _, _, _, WM_FOD_NORM_PATH, _, _) = get_mrtrix_fod_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Get the probabilistic tracking paths
    (GM_WM_SEED_PATH, TRACT_TCK_PATH) = get_mrtrix_probtrack_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    
    # Preparing mask for streamline seeding command
    GM_WM_SEED_CMD = "5tt2gmwmi {fivett_reg}.mif {output}.mif".format(fivett_reg=FIVETT_REG_PATH, output=GM_WM_SEED_PATH)
    # Probabilistic tractography command
    PROB_TRACT_CMD = "tckgen -act {fivett_reg}.mif -backtrack -seed_gmwmi {gmwm_seed}.mif -select 300000 \
        {wmfod_norm}.mif {output}.tck -algorithm iFOD2 -force".format(fivett_reg=FIVETT_REG_PATH, gmwm_seed=GM_WM_SEED_PATH,
            wmfod_norm=WM_FOD_NORM_PATH, output=TRACT_TCK_PATH)

    # Return the commands
    return (GM_WM_SEED_CMD, PROB_TRACT_CMD)

# Define MRtrix global tracking commands
def define_mrtrix_global_tracking_commands(ARGS):
        
    # Extract arguments needed to define paths
    SUBJECT_FILES = ARGS[0]
    MAIN_MRTRIX_PATH = ARGS[1]

    # Define what's needed for to extract from subject files
    FILES_NEEDED = ["filename", "dwi", "bval", "bvec"]
    NEEDED_FILE_PATHS = get_filename(SUBJECT_FILES, FILES_NEEDED)

    # Get the general paths
    (INPUT_MIF_PATH, MASK_MIF_PATH) = get_mrtrix_general_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Get the fod paths
    (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, 
        _, _, _, _, _, _, _, _) = get_mrtrix_fod_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Get the global tracking paths
    (GLOBAL_FOD_PATH, GLOBAL_FISO_PATH, GLOBAL_TRACT_PATH) = get_mrtrix_global_tracking_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    
    # Global tractography command
    GLOBAL_TRACT_CMD = "tckglobal {dwi}.mif {wm_response}.txt -riso {csf_response}.txt -riso \
         {gm_response}.txt -mask {mask}.mif -niter 1e9 -fod {gt_fod}.mif -fiso {gt_fiso}.mif {output}.tck".format(
            dwi=INPUT_MIF_PATH, wm_response=RESPONSE_WM_PATH, csf_response=RESPONSE_CSF_PATH, gm_response=RESPONSE_GM_PATH,
            mask=MASK_MIF_PATH, gt_fod=GLOBAL_FOD_PATH, gt_fiso=GLOBAL_FISO_PATH, output=GLOBAL_TRACT_PATH)

    # Return the commands
    return (GLOBAL_TRACT_CMD)

# Define MRtrix connectome commands
def define_mrtrix_connectome_commands(ARGS):
    
    # Extract arguments needed to define paths
    SUBJECT_FILES = ARGS[0]
    MAIN_MRTRIX_PATH = ARGS[1]
    ATLAS = ARGS[2]

    # Define what's needed for to extract from subject files
    FILES_NEEDED = ["filename", "dwi", "bval", "bvec"]
    NEEDED_FILE_PATHS = get_filename(SUBJECT_FILES, FILES_NEEDED)

    # Get the registration paths
    (_, _, _, _, _, _, _, _, _, _, ATLAS_REG_PATH, 
        _) = get_mrtrix_registration_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH, ATLAS)
    # Get the probabilistic tracking paths
    (_, TRACT_TCK_PATH) = get_mrtrix_probtrack_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Get the global tracking paths
    (_, _, GLOBAL_TRACT_PATH) = get_mrtrix_global_tracking_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    # Get the connectivity paths
    (CONNECTIVITY_PROB_PATH, CONNECTIVITY_GLOBAL_PATH) = get_mrtrix_connectome_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)
    
    # Connectivity matrix command
    CONNECTIVITY_PROB_CMD = "tck2connectome {input}.tck {atlas}.mif {output}.csv -zero_diagonal -symmetric \
        -assignment_all_voxels -force".format(input=TRACT_TCK_PATH, output=CONNECTIVITY_PROB_PATH, atlas=ATLAS_REG_PATH)
    CONNECTIVITY_GLOBAL_CMD = "tck2connectome {input}.tck {atlas}.mif {output}.csv -zero_diagonal -symmetric \
        -assignment_all_voxels -force".format(input=GLOBAL_TRACT_PATH, output=CONNECTIVITY_GLOBAL_PATH, atlas=ATLAS_REG_PATH)
    
    # Return the commands
    return (CONNECTIVITY_PROB_CMD, CONNECTIVITY_GLOBAL_CMD)

# Define MRTRIX commands
def probabilistic_tractography(ARGS):
    # Extract arguments needed to define paths
    SUBJECT_FILES = ARGS[0]
    CLEAN_FILES = ARGS[1]
    MAIN_MRTRIX_PATH = ARGS[2]
    STRIPPED_T1_PATH = ARGS[3]
    ATLAS = ARGS[4]

    # Define the FOD commands
    FOD_ARGS = [SUBJECT_FILES, CLEAN_FILES, MAIN_MRTRIX_PATH]
    (MIF_CMD, MASK_CMD, RESPONSE_EST_CMD, _, MULTISHELL_CSD_CMD, COMBINE_FODS_CMD,
        _, NORMALIZE_FODS_CMD, _) = define_mrtrix_fod_commands(FOD_ARGS)
    # Define the registration commands
    REG_ARGS = [SUBJECT_FILES, MAIN_MRTRIX_PATH, STRIPPED_T1_PATH, ATLAS]
    (MIF_T1_CMD, FIVETT_NOREG_CMD, DWI_B0_CMD, DWI_B0_NII_CMD, FIVETT_GEN_NII_CMD, REGISTER_T1_DWI_CMD,
        TRANSFORMATION_T1_DWI_CMD, FINAL_TRANSFORM_CMD, REGISTER_ATLAS_DWI_CMD, TRANSFORMATION_ATLAS_DWI_CMD,
            ATLAS_MIF_CMD, FINAL_ATLAS_TRANSFORM_CMD) = define_mrtrix_registration_commands(REG_ARGS)
    # Define the probabilistic tracking commands
    PROB_ARGS = [SUBJECT_FILES, MAIN_MRTRIX_PATH, ATLAS]
    (GM_WM_SEED_CMD, PROB_TRACT_CMD) = define_mrtrix_probtrack_commands(PROB_ARGS)
    # Define the global tracking commands
    GLOBAL_ARGS = [SUBJECT_FILES, MAIN_MRTRIX_PATH]
    (GLOBAL_TRACT_CMD) = define_mrtrix_global_tracking_commands(GLOBAL_ARGS)
    # Define the connectome commands
    CONNECTOME_ARGS = [SUBJECT_FILES, MAIN_MRTRIX_PATH, ATLAS]
    (CONNECTIVITY_PROB_CMD, CONNECTIVITY_GLOBAL_CMD) = define_mrtrix_connectome_commands(CONNECTOME_ARGS)


    # Create commands array
    MRTRIX_COMMANDS = [(MIF_CMD, "Convert DWI nii -> mif"), (MASK_CMD, "Create DWI brain mask"),
                        (RESPONSE_EST_CMD, "Estimate response function of WM, GM, CSF from DWI"),
                        (MULTISHELL_CSD_CMD, "Spherical deconvolution to estimate fODs"),
                        (COMBINE_FODS_CMD, "Combining fODs into a VF"),
                        (NORMALIZE_FODS_CMD, "Normalizing fODs"),
                        (MIF_T1_CMD, "Convert T1 nii -> mif"), (FIVETT_NOREG_CMD, "5ttgen creation with no registration"),
                        (DWI_B0_CMD, "Extracting mean B0 and transforming to NII"), (DWI_B0_NII_CMD, "DWI B0 mif -> NII"),
                        (FIVETT_GEN_NII_CMD, "5ttgen mif -> nii"), (REGISTER_T1_DWI_CMD, "Begin registering T1 to DWI space"),
                        (TRANSFORMATION_T1_DWI_CMD, "Initial transformation of T1 to DWI space"), 
                        (FINAL_TRANSFORM_CMD, "Final transformation of T1 to DWI space"),
                        (REGISTER_ATLAS_DWI_CMD, "Begin registering atlas to DWI space"),
                        (TRANSFORMATION_ATLAS_DWI_CMD, "Initial transformation of atlas to DWI space"),
                        (ATLAS_MIF_CMD, "Convert atlas nii -> mif"), (FINAL_ATLAS_TRANSFORM_CMD, "Final transformation of atlas to DWI space"),
                        (GM_WM_SEED_CMD, "Preparing mask for streamline seeding"), (PROB_TRACT_CMD, "Probabilistic tractography"),
                        (GLOBAL_TRACT_CMD, "Global tractography"), (CONNECTIVITY_PROB_CMD, "Creating connectivity matrix - probabilistic"),
                        (CONNECTIVITY_GLOBAL_CMD, "Creating connectivity matrix - global")]

    # Return the commands array
    return MRTRIX_COMMANDS