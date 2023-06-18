import os
import sys
import shutil
import glob

# -------------------------------------------------- MAIN FUNCTION MODULES -------------------------------------------------- #
def get_main_paths(hpc):
    # Depending on whether we're in HPC or not, paths change
    if hpc == True:
        DWI_MAIN_FOLDER = "/rds/general/user/hsa22/home/dissertation/tractography"
        DWI_OUTPUT_FOLDER = "/rds/general/user/hsa22/home/dissertation/tractography/dsi_outputs"
        DWI_LOGS_FOLDER = "/rds/general/user/hsa22/home/dissertation/tractography/logs"

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
            DWI_MAIN_FOLDER = os.path.realpath(os.path.join(os.getcwd(), "data/subjects"))
            DWI_OUTPUT_FOLDER = os.path.realpath(os.path.join(os.getcwd(), "data/dsi_outputs"))
            DWI_LOGS_FOLDER = os.path.realpath(os.path.join(os.getcwd(), "data/logs"))

            DSI_COMMAND = "dsi_studio"

            ATLAS_FOLDER = os.path.realpath(os.path.join(os.getcwd(), "data/atlas"))
            TRACT_FOLDER = os.path.realpath(os.path.join(os.getcwd(), "data/tracts"))

    MAIN_STUDIO_PATH = os.path.join(DWI_OUTPUT_FOLDER, "studio")
    MAIN_MRTRIX_PATH = os.path.join(DWI_OUTPUT_FOLDER, "mrtrix")

    # Return folder names
    return (DWI_MAIN_FOLDER, DWI_OUTPUT_FOLDER, DWI_LOGS_FOLDER, DSI_COMMAND, ATLAS_FOLDER, TRACT_FOLDER, 
            MAIN_STUDIO_PATH, MAIN_MRTRIX_PATH)

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

# -------------------------------------------------- PARALLEL FUNCTION MODULES -------------------------------------------------- #
# Get name of DWI file
def get_dwi_filename(DWI_INPUT_FILE):
    # Get filename based on Windows or Linux
    if os.name == 'nt':
        dwi_filename = DWI_INPUT_FILE.split("\\")[-1]
    else:
        dwi_filename = DWI_INPUT_FILE.split("/")[-1]
    # Remove extension to get name only
    dwi_filename = dwi_filename.replace(".nii.gz", "")

    return dwi_filename

# Get the DSI_STUDIO file paths
def get_dsi_studio_paths(dwi_filename, MAIN_STUDIO_PATH, DWI_LOGS_FOLDER):
    STUDIO_SRC_PATH = os.path.join(MAIN_STUDIO_PATH, "{}_clean".format(dwi_filename))
    STUDIO_DTI_PATH = os.path.join(MAIN_STUDIO_PATH, "{}_dti".format(dwi_filename))
    STUDIO_QSDR_PATH = os.path.join(MAIN_STUDIO_PATH, "{}_qsdr".format(dwi_filename))

    SRC_LOG_PATH = os.path.join(DWI_LOGS_FOLDER, os.path.join("dwi_to_src", "src_log_{}.txt".format(dwi_filename)))
    DTI_LOG_PATH = os.path.join(DWI_LOGS_FOLDER, os.path.join("src_to_dti","dti_log_{}.txt".format(dwi_filename)))
    DTI_EXP_LOG_PATH = os.path.join(DWI_LOGS_FOLDER, os.path.join("dti_export","exporting_dti_log_{}.txt".format(dwi_filename)))
    QSDR_LOG_PATH = os.path.join(DWI_LOGS_FOLDER, os.path.join("src_to_qsdr","qsdr_log_{}.txt".format(dwi_filename)))
    QSDR_EXP_LOG_PATH = os.path.join(DWI_LOGS_FOLDER, os.path.join("qsdr_export","exporting_qsdr_log_{}.txt".format(dwi_filename)))
    TRACT_LOG_PATH = os.path.join(DWI_LOGS_FOLDER, os.path.join("recon_to_tract","tract_log_{}.txt".format(dwi_filename)))

    # Return the paths
    return (STUDIO_SRC_PATH, STUDIO_DTI_PATH, STUDIO_QSDR_PATH, SRC_LOG_PATH, DTI_LOG_PATH, DTI_EXP_LOG_PATH,
            QSDR_LOG_PATH, QSDR_EXP_LOG_PATH, TRACT_LOG_PATH)

# Define DSI_STUDIO commands
def define_studio_commands(ARGS):
    # Extract arguments needed to define commands
    MAIN_STUDIO_PATH = ARGS[0]
    DWI_LOGS_FOLDER = ARGS[1]

    DSI_COMMAND = ARGS[2]
    DWI_INPUT_FILE = ARGS[3]
    B_VAL_FILE = ARGS[4]
    B_VEC_FILE = ARGS[5]
    ATLAS_STRING = ARGS[6]

    # Get the rest of the paths for the commands
    dwi_filename = get_dwi_filename(DWI_INPUT_FILE)
    (SRC_PATH, DTI_PATH, QSDR_PATH, SRC_LOG_PATH, DTI_LOG_PATH, DTI_EXP_LOG_PATH,
            QSDR_LOG_PATH, QSDR_EXP_LOG_PATH, TRACT_LOG_PATH) = get_dsi_studio_paths(dwi_filename, MAIN_STUDIO_PATH, DWI_LOGS_FOLDER)
    
    # Define the commands
    SRC_CMD = "{} --action=src --source={} --bval={} --bvec={} --output={} > {}".format(DSI_COMMAND,
                                                    DWI_INPUT_FILE, B_VAL_FILE, B_VEC_FILE, SRC_PATH, SRC_LOG_PATH)
    DTI_CMD = "{} --action=rec --source={}.src.gz --method=1 --record_odf=1 \
        --param0=1.25 --motion_correction=0 --output={}.fib.gz > {}".format(DSI_COMMAND, SRC_PATH, DTI_PATH, DTI_LOG_PATH)
    EXPORT_DTI_CMD = "{} --action=exp --source={}.fib.gz --export=fa > {}".format(DSI_COMMAND, DTI_PATH, DTI_EXP_LOG_PATH)
    QSDR_CMD = "{} --action=rec --source={}.src.gz --method=7 --record_odf=1 \
        --param0=1.25 --motion_correction=0 --other_image=fa:{}.fib.gz.fa.nii.gz --output={}.fib.gz \
            > {}".format(DSI_COMMAND, SRC_PATH, DTI_PATH, QSDR_PATH, QSDR_LOG_PATH)
    EXPORT_QSDR_CMD = "{} --action=exp --source={}.fib.gz --export=qa,rdi,fa,md > {}".format(DSI_COMMAND, QSDR_PATH, QSDR_EXP_LOG_PATH)

    STUDIO_DET_TRACT_CMD = "{} --action=trk --source={}.fib.gz --fiber_count=1000000 --output=no_file \
    --method=0 --interpolation=0 --max_length=400 --min_length=10 --otsu_threshold=0.6 --random_seed=0 --turning_angle=55 \
        --smoothing=0 --step_size=1 --connectivity={} --connectivity_type=end \
            --connectivity_value=count --connectivity_threshold=0.001 > {}".format(DSI_COMMAND, QSDR_PATH, ATLAS_STRING, TRACT_LOG_PATH)

    # Create commands array
    STUDIO_COMMANDS = [(SRC_CMD, "SRC creation"), (DTI_CMD, "DTI evaluation"), 
                       (EXPORT_DTI_CMD, "Exporting DTI metrics"), (QSDR_CMD, "QSDR evaluation"), 
                       (EXPORT_QSDR_CMD, "Exporting QSDR metrics"), (STUDIO_DET_TRACT_CMD, "Deterministic tractography")]

    # Return the commands array
    return STUDIO_COMMANDS

# Get the MRTRIX file paths
def get_mrtrix_paths(dwi_filename, MAIN_MRTRIX_PATH):
    MRTRIX_MIF_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}".format(dwi_filename))

    return MRTRIX_MIF_PATH

# Define MRTRIX commands
def define_mrtrix_commands(ARGS):
    # Extract arguments needed to define paths
    DWI_INPUT_FILE = ARGS[0]
    B_VEC_FILE = ARGS[1]
    B_VAL_FILE = ARGS[2]
    MAIN_MRTRIX_PATH = ARGS[3]

    # Get the rest of the paths for the commands
    dwi_filename = get_dwi_filename(DWI_INPUT_FILE)
    MRTRIX_MIF_PATH = get_mrtrix_paths(dwi_filename, MAIN_MRTRIX_PATH)

    # Define the commands
    MRTRIX_MIF_CMD = "mrconvert {} -fslgrad {} {} {}.mif".format(DWI_INPUT_FILE, B_VEC_FILE, B_VAL_FILE, MRTRIX_MIF_PATH)

    # Create commands array
    MRTRIX_COMMANDS = [(MRTRIX_MIF_CMD, "Conversion NifTI -> MIF")]

    # Return the commands array
    return MRTRIX_COMMANDS

