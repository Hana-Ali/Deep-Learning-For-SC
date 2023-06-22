import os
import sys
import shutil
import glob
import regex as re

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
            DWI_MAIN_FOLDER = os.path.realpath(r"C:\\tractography\\subjects")
            DWI_OUTPUT_FOLDER = os.path.realpath(r"C:\\tractography\\dsi_outputs")
            DWI_LOGS_FOLDER = os.path.realpath(r"C:\\tractography\\logs")

            DSI_COMMAND = "dsi_studio"

            ATLAS_FOLDER = os.path.realpath(r"C:\\tractography\\atlas")
            TRACT_FOLDER = os.path.realpath(r"C:\\tractography\\tracts")
        else:
            DWI_MAIN_FOLDER = os.path.realpath(os.path.join(os.getcwd(), "data/subjects/dwi"))
            T1_MAIN_FOLDER = os.path.realpath(os.path.join(os.getcwd(), "data/subjects/t1"))
            DWI_OUTPUT_FOLDER = os.path.realpath(os.path.join(os.getcwd(), "data/dsi_outputs"))

            DSI_COMMAND = "dsi_studio"

            ATLAS_FOLDER = os.path.realpath(os.path.join(os.getcwd(), "data/atlas"))
            TRACT_FOLDER = os.path.realpath(os.path.join(os.getcwd(), "data/tracts"))

    MAIN_STUDIO_PATH = os.path.join(DWI_OUTPUT_FOLDER, "studio")
    MAIN_MRTRIX_PATH = os.path.join(DWI_OUTPUT_FOLDER, "mrtrix")
    MAIN_FSL_PATH = os.path.join(DWI_OUTPUT_FOLDER, "fsl")

    # Return folder names
    return (DWI_MAIN_FOLDER, T1_MAIN_FOLDER, DWI_OUTPUT_FOLDER, DSI_COMMAND, ATLAS_FOLDER, 
            TRACT_FOLDER, MAIN_STUDIO_PATH, MAIN_MRTRIX_PATH, MAIN_FSL_PATH)

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

# Function to extract the DWI filename
def extract_dwi_filename(dwi_file):
    # Extract the filename
    dwi_filename = dwi_file.split("/sub-")[-1].split("_")[0]
    # Return the filename
    return dwi_filename

# Function to extract the T1 filename
def extract_t1_filename(t1_file):
    # Extract the filename
    t1_filename = t1_file.split("/")[-1].split("_")[1].split("-")[0]
    # Return the filename
    return t1_filename

# Create a list that associates each subject with its T1 and DWI files
def create_subject_list(DWI_INPUT_FILES, DWI_JSON_HEADERS, B_VAL_FILES, B_VEC_FILES, T1_INPUT_FILES):
    SUBJECT_LISTS = []
    DWI_LIST = []
    JSON_LIST = []
    B_VAL_LIST = []
    B_VEC_LIST = []
    T1_LIST = []

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

    # Join the two lists based on common subject name
    for dwi in DWI_LIST:
        # Get the name, or common element ID
        dwi_name = dwi[0]
        # Based on this name, get every json, bval, bvec and t1 that has the same name
        json = [json[1] for json in JSON_LIST if json[0] == dwi_name]
        bval = [bval[1] for bval in B_VAL_LIST if bval[0] == dwi_name]
        bvec = [bvec[1] for bvec in B_VEC_LIST if bvec[0] == dwi_name]
        t1 = [t1[1] for t1 in T1_LIST if t1[0] == dwi_name]
        # Append the subject name, dwi, json, bval, bvec and t1 to the list
        SUBJECT_LISTS.append([dwi_name, dwi[1], json[0], bval[0], bvec[0], t1[0]])
        
    return SUBJECT_LISTS

# -------------------------------------------------- PARALLEL FUNCTION MODULES -------------------------------------------------- #

# Get filename
def get_filename(SUBJECT_FILES, items):
    # Define the indices of the subject list
    SUBJECT_DWI_NAME = 0
    SUBJECT_DWI_INDEX = 1
    SUBJECT_DWI_JSON_INDEX = 2
    SUBJECT_BVAL_INDEX = 3
    SUBJECT_BVEC_INDEX = 4
    SUBJECT_T1_INDEX = 5
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
        else:
            print("Item {} not found".format(item))
            sys.exit('Exiting program')
            
    return items_to_get

# Get the FSL file paths
def get_fsl_paths(NEEDED_FILE_PATHS, MAIN_FSL_PATH):

    # Extract what we need here from the needed file paths
    filename = NEEDED_FILE_PATHS["filename"]

    # Creating folder for each subject in FSL folder
    SUBJECT_FOLDER_NAME = os.path.join(MAIN_FSL_PATH, filename)
    check_output_folders(SUBJECT_FOLDER_NAME, "FSL subject folder")
    
    # Define the path for skull stripped T1
    SKULL_STRIP_PATH = os.path.join(SUBJECT_FOLDER_NAME, "_skull_strip")

    return (SKULL_STRIP_PATH)

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
    check_output_folders(CLEANING_FOLDER_NAME, "MRtrix cleaning folder")

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
    DWI_CONVERT_CMD = "mrconvert {input}.mif {output}.nii -export_grad_fsl {bvec_clean} {bval_clean}".format(
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

# Get the DSI_STUDIO file paths
def get_dsi_studio_paths(NEEDED_FILE_PATHS, MAIN_STUDIO_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]

    # Creating folder for each subject in DSI_STUDIO folder. CAN WIPE as this is the only function that uses it
    SUBJECT_FOLDER_NAME = os.path.join(MAIN_STUDIO_PATH, dwi_filename)
    check_output_folders(SUBJECT_FOLDER_NAME, "DSI Studio subject folder")
    # Creating logs folder for each subject. CAN WIPE as this is the only function that uses it
    SUBJECT_LOGS_FOLDER = os.path.join(SUBJECT_FOLDER_NAME, "logs")
    check_output_folders(SUBJECT_LOGS_FOLDER, "DSI Studio logs folder")

    # DWI nii -> src filepath
    STUDIO_SRC_PATH = os.path.join(SUBJECT_FOLDER_NAME, "{}_clean".format(dwi_filename))
    # DTI estimation filepath
    STUDIO_DTI_PATH = os.path.join(SUBJECT_FOLDER_NAME, "{}_dti".format(dwi_filename))
    # QSDR estimation filepath
    STUDIO_QSDR_PATH = os.path.join(SUBJECT_FOLDER_NAME, "{}_qsdr".format(dwi_filename))

    # Log file paths
    SRC_LOG_PATH = os.path.join(SUBJECT_LOGS_FOLDER, os.path.join("dwi_to_src", "src_log_{}.txt".format(dwi_filename)))
    DTI_LOG_PATH = os.path.join(SUBJECT_LOGS_FOLDER, os.path.join("src_to_dti","dti_log_{}.txt".format(dwi_filename)))
    DTI_EXP_LOG_PATH = os.path.join(SUBJECT_LOGS_FOLDER, os.path.join("dti_export","exporting_dti_log_{}.txt".format(dwi_filename)))
    QSDR_LOG_PATH = os.path.join(SUBJECT_LOGS_FOLDER, os.path.join("src_to_qsdr","qsdr_log_{}.txt".format(dwi_filename)))
    QSDR_EXP_LOG_PATH = os.path.join(SUBJECT_LOGS_FOLDER, os.path.join("qsdr_export","exporting_qsdr_log_{}.txt".format(dwi_filename)))
    TRACT_LOG_PATH = os.path.join(SUBJECT_LOGS_FOLDER, os.path.join("recon_to_tract","tract_log_{}.txt".format(dwi_filename)))

    # Return the paths
    return (STUDIO_SRC_PATH, STUDIO_DTI_PATH, STUDIO_QSDR_PATH, SRC_LOG_PATH, DTI_LOG_PATH, 
            DTI_EXP_LOG_PATH, QSDR_LOG_PATH, QSDR_EXP_LOG_PATH, TRACT_LOG_PATH)

# Define DSI_STUDIO commands
def define_studio_commands(ARGS):
    # Extract arguments needed to define commands
    SUBJECT_FILES = ARGS[0]
    CLEAN_FILES = ARGS[1]
    MAIN_STUDIO_PATH = ARGS[2]
    DSI_COMMAND = ARGS[3]
    ATLAS_STRING = ARGS[4]
    
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
        --smoothing=0 --step_size=1 --connectivity={atlas_string} --connectivity_type=end \
            --connectivity_value=count --connectivity_threshold=0.001 > {log}".format(
        command=DSI_COMMAND, qsdr=QSDR_PATH, atlas_string=ATLAS_STRING, log=TRACT_LOG_PATH)

    # Create commands array
    STUDIO_COMMANDS = [(SRC_CMD, "SRC creation"), (DTI_CMD, "DTI evaluation"), 
                       (EXPORT_DTI_CMD, "Exporting DTI metrics"), (QSDR_CMD, "QSDR evaluation"), 
                       (EXPORT_QSDR_CMD, "Exporting QSDR metrics"), (STUDIO_DET_TRACT_CMD, "Deterministic tractography")]

    # Return the commands array
    return STUDIO_COMMANDS

# Get the MRTRIX file paths
def get_mrtrix_recon_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH):

    # Extract what we need here from the needed file paths
    dwi_filename = NEEDED_FILE_PATHS["filename"]

    # Creating folder for each subject in MRTRIX folder. DON'T WIPE as it'll be used by other functions
    SUBJECT_FOLDER_NAME = os.path.join(MAIN_MRTRIX_PATH, dwi_filename)
    if not os.path.exists(SUBJECT_FOLDER_NAME):
        print("--- MRtrix subject folder not found. Created folder: {}".format(SUBJECT_FOLDER_NAME))
        os.makedirs(SUBJECT_FOLDER_NAME)
    
    # Creating folder for cleaning in MRTRIX folder. WIPE since this is the only function that uses it
    RECON_FOLDER_NAME = os.path.join(SUBJECT_FOLDER_NAME, "reconstruction")
    check_output_folders(RECON_FOLDER_NAME, "MRtrix reconstruction folder")

    # Because there's a lot of commands, we define extra folder paths here
    RESPONSE_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "response")
    FOD_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "fod")
    FOD_NORM_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "fod_norm")
    T1_REG_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "t1_and_Fivett_reg")
    TRACKING_FOLDER_NAME = os.path.join(RECON_FOLDER_NAME, "tracking")
    
    # DWI nii -> mif filepath
    INPUT_MIF_PATH = os.path.join(RECON_FOLDER_NAME, "{}_recon_input".format(dwi_filename))
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
    # Path of the brain mask
    MASK_MIF_PATH = os.path.join(T1_REG_FOLDER_NAME, "{}_mask".format(dwi_filename))
    # T1 nii -> mif filepath
    T1_MIF_PATH = os.path.join(T1_REG_FOLDER_NAME, "{}_t1".format(dwi_filename))
    # 5ttgen without coregistration filepath
    FIVETT_GEN_PATH = os.path.join(T1_REG_FOLDER_NAME, "{}_5ttgen".format(dwi_filename))
    # Mean B0 of DWI filepath
    DWI_B0_PATH = os.path.join(T1_REG_FOLDER_NAME, "{}_b0".format(dwi_filename))
    # Mean B0 mif -> nii filepath
    DWI_B0_NII = os.path.join(T1_REG_FOLDER_NAME, "{}_b0.nii.gz".format(dwi_filename))
    # 5ttgen mif -> nii filepath
    FIVETT_GEN_NII = os.path.join(T1_REG_FOLDER_NAME, "{}_5ttgen.nii.gz".format(dwi_filename))
    # Registerting and transforming T1 to DWI space paths
    T1_DWI_MAP_MAT = os.path.join(T1_REG_FOLDER_NAME, "{}_t12dwi_fsl".format(dwi_filename))
    T1_DWI_CONVERT_INV = os.path.join(T1_REG_FOLDER_NAME, "{}_t12dwi_mrtrix".format(dwi_filename))
    FIVETT_REG_PATH = os.path.join(T1_REG_FOLDER_NAME, "{}_5ttgenreg".format(dwi_filename))
    # Mask for streamline seeding paths
    GM_WM_SEED_PATH = os.path.join(TRACKING_FOLDER_NAME, "{}_gmwmseed".format(dwi_filename))
    # Fiber tracking paths
    TRACT_TCK_PATH = os.path.join(TRACKING_FOLDER_NAME, "{}_tract".format(dwi_filename))
    CONNECTIVITY_PATH = os.path.join(TRACKING_FOLDER_NAME, "{}_connectivity".format(dwi_filename))

    return (INPUT_MIF_PATH, RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH, WM_FOD_PATH,
            GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, CSF_FOD_NORM_PATH,
            MASK_MIF_PATH, T1_MIF_PATH, FIVETT_GEN_PATH, DWI_B0_PATH, DWI_B0_NII, FIVETT_GEN_NII, T1_DWI_MAP_MAT,
            T1_DWI_CONVERT_INV, FIVETT_REG_PATH, GM_WM_SEED_PATH, TRACT_TCK_PATH, CONNECTIVITY_PATH)

# Define MRTRIX commands
def define_mrtrix_recon_commands(ARGS):
    # Extract arguments needed to define paths
    SUBJECT_FILES = ARGS[0]
    CLEAN_FILES = ARGS[1]
    MAIN_MRTRIX_PATH = ARGS[2]
    STRIPPED_T1_PATH = ARGS[3]
    ATLAS = ARGS[4]

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
    (INPUT_MIF_PATH, RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH, WM_FOD_PATH,
    GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, CSF_FOD_NORM_PATH,
    MASK_MIF_PATH, T1_MIF_PATH, FIVETT_GEN_PATH, DWI_B0_PATH, DWI_B0_NII, FIVETT_GEN_NII, T1_DWI_MAP_MAT,
    T1_DWI_CONVERT_INV, FIVETT_REG_PATH, GM_WM_SEED_PATH, TRACT_TCK_PATH, 
    CONNECTIVITY_PATH) = get_mrtrix_recon_paths(NEEDED_FILE_PATHS, MAIN_MRTRIX_PATH)

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
    # T1 nii -> mif command
    MIF_T1_CMD = "mrconvert {input_nii} {output}.mif".format(input_nii=STRIPPED_T1_PATH, output=T1_MIF_PATH)
    # 5ttgen no reg command
    FIVETT_GEN_CMD = "5ttgen fsl {input}.mif {output}.mif".format(input=T1_MIF_PATH, output=FIVETT_GEN_PATH)
    # Extracting the mean B0 from DWI command
    DWI_B0_CMD = "dwiextract {input}.mif - -bzero | mrmath - mean {output}.mif -axis 3".format(
        input=INPUT_MIF_PATH, output=DWI_B0_PATH)
    # Mean B0 mif -> nii conversion command
    DWI_B0_NII_CMD = "mrconvert {input}.mif {output}".format(input=DWI_B0_PATH, output=DWI_B0_NII)
    # 5ttgen mif -> nii command
    FIVETT_GEN_NII_CMD = "mrconvert {input}.mif {output}".format(input=FIVETT_GEN_PATH, output=FIVETT_GEN_NII)
    # Transformation and registration of T1 to DWI space commands
    REGISTER_T1_DWI_CMD = "flirt -in {dwi} -ref {fivett} -interp nearestneighbour -dof 6 -omat {transform_mat}.mat".format(
        dwi=DWI_B0_NII, fivett=FIVETT_GEN_NII, transform_mat=T1_DWI_MAP_MAT)
    TRANSFORMATION_T1_DWI_CMD = "transformconvert {transform_mat}.mat {dwi} {fivett} flirt_import {output}.txt".format(
        transform_mat=T1_DWI_MAP_MAT, dwi=DWI_B0_NII, fivett=FIVETT_GEN_NII, output=T1_DWI_CONVERT_INV)
    FINAL_TRANSFORM_CMD = "mrtransform {fivett}.mif -linear {transform}.txt -inverse {output}.mif".format(
        fivett=FIVETT_GEN_PATH, transform=T1_DWI_CONVERT_INV, output=FIVETT_REG_PATH)

    # Preparing mask for streamline seeding command
    MASK_SEEDING_CMD = "5tt2gmwmi {fivett_reg}.mif {output}.mif".format(fivett_reg=FIVETT_REG_PATH, output=GM_WM_SEED_PATH)

    PROB_TRACT_CMD = "tckgen -act {fivett_reg} -backtrack -seed_gmwmi {gmwm_seed}.mif -select 300000 \
          {wmfod_norm}.mif {output}.tck -algorithm iFOD2 -force".format( fivett_reg=FIVETT_REG_PATH, gmwm_seed=GM_WM_SEED_PATH,
        wmfod_norm=WM_FOD_NORM_PATH, output=TRACT_TCK_PATH)
    CONNECTIVITY_CMD = "tck2connectome {input}.tck {atlas} {output}.csv -zero_diagonal -symmetric \
        -assignment_all_voxels -force".format(
        input=TRACT_TCK_PATH, output=CONNECTIVITY_PATH, atlas=ATLAS)
    
    # Create commands array
    MRTRIX_COMMANDS = [(MIF_CMD, "Conversion NifTI -> MIF"), (MASK_CMD, "Brain mask creation"),
                       (RESPONSE_EST_CMD, "Response estimation"),
                       (MULTISHELL_CSD_CMD, "Spherical deconvolution"), (COMBINE_FODS_CMD, "Combining fODs"),
                       (NORMALIZE_FODS_CMD, "Normalizing fODs"),
                       (MIF_T1_CMD, "Conversion T1 NifTI -> MIF"),
                       (FIVETT_GEN_CMD, "5ttgen"), (DWI_B0_CMD, "Extracting mean B0 from DWI"),
                       (DWI_B0_NII_CMD, "Conversion mean B0 MIF -> NifTI"), (FIVETT_GEN_NII_CMD, "Conversion 5ttgen MIF -> NifTI"),
                       (REGISTER_T1_DWI_CMD, "Registering T1 to DWI space"), (TRANSFORMATION_T1_DWI_CMD, "Transformation T1 to DWI space"),
                       (FINAL_TRANSFORM_CMD, "Final transformation"), (MASK_SEEDING_CMD, "Preparing mask for streamline seeding"),
                       (PROB_TRACT_CMD, "Probabilistic tractography"), (CONNECTIVITY_CMD, "Connectivity matrix")]

    # Return the commands array
    return MRTRIX_COMMANDS


# REGISTER_T1_DWI_CMD = "flirt -in {input} -ref {ref} -out {output} -omat {output_mat} -dof 6 -cost mutualinfo -searchcost mutualinfo".format(
#     input=DWI_B0_NII, ref=FIVETT_GEN_NII, output=FIVETT_GEN_NII, output_mat=os.path.join(MAIN_MRTRIX_PATH, "t1_to_dwi.mat"))
# CHECK_QSPACE_SAMPLE = "mrinfo {}.mif -dwgrad".format(MIF_PATH)
# CHECK_QSPACE_SAMPLE2 = "mrinfo -shell_bvalues {}.mif".format(MIF_PATH)
# CHECK_QSPACE_SAMPLE3 = "mrinfo -shell_sizes {}.mif".format(MIF_PATH)
# inspect voxels used for response function estimation
# INSPECT_RESPONSE_CMD = "mrview {input_nii} -overlay.load {response_voxels}.mif -plane 2".format(input_nii=DWI_INPUT_FILE,
#                                                                                                 response_voxels=RESPONSE_VOXEL_PATH)
# inspect response functions
# INSPECT_RESPONSE_CMD2 = "shview {wm}.txt {gm}.txt {csf}.txt".format(wm=RESPONSE_WM_PATH, gm=RESPONSE_GM_PATH, csf=RESPONSE_CSF_PATH)
# Create commands array

# PROB_TRACT_CMD = "tckgen {input}.mif {output}.tck -act {wm}.mif -backtrack -crop_at_gmwmi -seed_dynamic {input}.mif -maxlength 250 -select 100000 \
#     -cutoff 0.06 -minlength 10 -power 1.0 -nthreads 0".format(
#         input=INPUT_MIF_PATH, output=TRACT_MIF_PATH, wm=RESPONSE_WM_PATH)