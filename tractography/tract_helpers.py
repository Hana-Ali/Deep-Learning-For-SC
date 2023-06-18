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
    INPUT_MIF_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}".format(dwi_filename))
    RESPONSE_WM_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_wm".format(dwi_filename))
    RESPONSE_GM_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_gm".format(dwi_filename))
    RESPONSE_CSF_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_csf".format(dwi_filename))
    RESPONSE_VOXEL_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_response_voxels".format(dwi_filename))
    WM_FOD_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_wmfod".format(dwi_filename))
    GM_FOD_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_gmfod".format(dwi_filename))
    CSF_FOD_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_csffod".format(dwi_filename))
    VF_FOD_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_vf".format(dwi_filename))
    WM_FOD_NORM_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_wmfod_norm".format(dwi_filename))
    GM_FOD_NORM_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_gmfod_norm".format(dwi_filename))
    CSF_FOD_NORM_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_csffod_norm".format(dwi_filename))
    MASK_MIF_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_mask".format(dwi_filename))
    TRACT_TCK_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_tract".format(dwi_filename))
    CONNECTIVITY_PATH = os.path.join(MAIN_MRTRIX_PATH, "{}_connectivity".format(dwi_filename))

    return (INPUT_MIF_PATH, RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH, WM_FOD_PATH,
            GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, CSF_FOD_NORM_PATH,
            MASK_MIF_PATH, TRACT_TCK_PATH, CONNECTIVITY_PATH)

# Define MRTRIX commands
def define_mrtrix_commands(ARGS):
    # Extract arguments needed to define paths
    DWI_INPUT_FILE = ARGS[0]
    B_VEC_FILE = ARGS[1]
    B_VAL_FILE = ARGS[2]
    MAIN_MRTRIX_PATH = ARGS[3]
    ATLAS = ARGS[4]

    # Get the rest of the paths for the commands
    dwi_filename = get_dwi_filename(DWI_INPUT_FILE)
    (INPUT_MIF_PATH, RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH, WM_FOD_PATH,
        GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, CSF_FOD_NORM_PATH,
            MASK_MIF_PATH, TRACT_TCK_PATH, CONNECTIVITY_PATH) = get_mrtrix_paths(dwi_filename, MAIN_MRTRIX_PATH)

    # Define the commands
    MIF_CMD = "mrconvert {input_nii} -fslgrad {bvec} {bval} {output}.mif".format(input_nii=DWI_INPUT_FILE, 
                                                                                bvec=B_VEC_FILE, 
                                                                                bval=B_VAL_FILE, 
                                                                                output=INPUT_MIF_PATH)
    RESPONSE_EST_CMD = "dwi2response dhollander {input}.mif {wm}.txt {gm}.txt {csf}.txt -voxels {response_voxels}.mif".format(
        input=INPUT_MIF_PATH, wm=RESPONSE_WM_PATH, gm=RESPONSE_GM_PATH, csf=RESPONSE_CSF_PATH, response_voxels=RESPONSE_VOXEL_PATH)
    VIEW_RESPONSE_CMD = "mrview {input}.mif -overlay.load {response_voxels}.mif".format(input=INPUT_MIF_PATH,
                                                                                        response_voxels=RESPONSE_VOXEL_PATH)    
    MASK_CMD = "dwi2mask {input}.mif {output}.mif".format(input=INPUT_MIF_PATH, output=MASK_MIF_PATH)
    MULTISHELL_CSD_CMD = "dwi2fod msmt_csd {input}.mif {wm}.txt {wmfod}.mif {gm}.txt {gmfod}.mif {csf}.txt \
        {csffod}.mif -mask {mask}.mif".format(
        input=INPUT_MIF_PATH, wm=RESPONSE_WM_PATH, wmfod=WM_FOD_PATH, gm=RESPONSE_GM_PATH, gmfod=GM_FOD_PATH,
        csf=RESPONSE_CSF_PATH, csffod=CSF_FOD_PATH, mask=MASK_MIF_PATH)
    COMBINE_FODS_CMD = "mrconvert -coord 3 0 {wmfod}.mif - | mrcat {csffod}.mif {gmfod}.mif - {output}.mif".format(
        wmfod=WM_FOD_PATH, csffod=CSF_FOD_PATH, gmfod=GM_FOD_PATH, output=VF_FOD_PATH)
    VIEW_COMBINED_FODS_CMD = "mrview {vf}.mif -odf.load_sh {wmfod}.mif".format(vf=VF_FOD_PATH, wmfod=WM_FOD_PATH)
    NORMALIZE_FODS_CMD = "mtnormalise {wmfod}.mif {wmfod_norm}.mif {gmfod}.mif {gmfod_norm}.mif {csffod}.mif \
        {csffod_norm}.mif -mas {mask}.mif".format(
        wmfod=WM_FOD_PATH, wmfod_norm=WM_FOD_NORM_PATH, gmfod=GM_FOD_PATH, gmfod_norm=GM_FOD_NORM_PATH, csffod=CSF_FOD_PATH,
        csffod_norm=CSF_FOD_NORM_PATH, mask=MASK_MIF_PATH)
    VIEW_NORMALIZED_FODS_CMD = "mrview {vf}.mif -odf.load_sh {wmfod_norm}.mif".format(vf=VF_FOD_PATH,
                                                                                      wmfod_norm=WM_FOD_NORM_PATH)
    ##### NEED TO DO 5TTGEN FIRST
    PROB_TRACT_CMD = "tckgen {wmfod_norm}.mif {output}.tck -seed_image {seed_image} -mask {mask} \
        -algorithm iFOD2 -select 300000 -force".format(
        wmfod_norm=WM_FOD_NORM_PATH, output=TRACT_TCK_PATH, seed_image=ATLAS, mask=ATLAS)
    CONNECTIVITY_CMD = "tck2connectome {input}.tck {atlas} {output}.csv -zero_diagonal -symmetric \
        -assignment_all_voxels -force".format(
        input=TRACT_TCK_PATH, output=CONNECTIVITY_PATH, atlas=ATLAS)
    
    # Create commands array
    MRTRIX_COMMANDS = [#(MIF_CMD, "Conversion NifTI -> MIF"), (RESPONSE_EST_CMD, "Response function estimation"),
                       #(MASK_CMD, "Mask creation"), (MULTISHELL_CSD_CMD, "Multi-shell CSD"),
                       #(COMBINE_FODS_CMD, "Combining FODs"), (NORMALIZE_FODS_CMD, "Normalizing FODs"),
                       (PROB_TRACT_CMD, "Probabilistic tractography"), (CONNECTIVITY_CMD, "Connectivity matrix")]

    # Return the commands array
    return MRTRIX_COMMANDS

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