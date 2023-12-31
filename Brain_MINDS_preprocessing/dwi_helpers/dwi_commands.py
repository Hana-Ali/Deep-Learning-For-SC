from .dwi_paths import *
from .dwi_checkpoints import *

from py_helpers import *

# Function to get a clean image as reference from the BMINDS clean
def get_clean_reference(hpc):

    # Get all the main paths
    (BMINDS_DATA_FOLDER, BMINDS_OUTPUTS_DMRI_FOLDER, BMINDS_OUTPUTS_DMRI_BMA_FOLDER, BMINDS_OUTPUTS_INVIVO_BMA_FOLDER,
    BMINDS_OUTPUTS_EXVIVO_BMA_FOLDER, BMINDS_OUTPUTS_INJECTIONS_FOLDER, BMINDS_BMA_MAIN_FOLDER,
    BMINDS_BMA_INVIVO_FOLDER, BMINDS_BMA_EXVIVO_FOLDER, BMINDS_BMA_INVIVO_DWI_FOLDER, BMINDS_BMA_EXVIVO_DWI_FOLDER,
    BMINDS_CORE_FOLDER, BMINDS_DWI_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_TEMPLATES_FOLDER,
    BMINDS_ATLAS_FOLDER, BMINDS_STPT_TEMPLATE_FOLDER, BMINDS_TRANSFORMS_FOLDER, BMINDS_INJECTIONS_FOLDER,
    BMINDS_UNZIPPED_DWI_FOLDER, BMINDS_UNZIPPED_DWI_RESIZED_FOLDER,
    MAIN_MRTRIX_FOLDER_DMRI, MAIN_MRTRIX_FOLDER_BMA_DMRI_INVIVO, MAIN_MRTRIX_FOLDER_BMA_DMRI_EXVIVO,
    MAIN_MRTRIX_FOLDER_INJECTIONS) = get_main_paths(hpc=hpc)

    # Get all the folder names
    BMINDS_MRTRIX_FOLDER_NAMES = [file for file in os.listdir(MAIN_MRTRIX_FOLDER_DMRI) if "A9" not in file]

    # Choose a random folder
    random_folder = np.random.choice(BMINDS_MRTRIX_FOLDER_NAMES)

    # Get the general folder name and the specific folder name
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
    FOD_NORM_FOLDER_NAME, REGISTRATION_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
    CONNECTIVITY_FOLDER_NAME) = define_main_mrtrix_folders_for_each_data("BMCR", random_folder)

    # Get all nii files in the folder
    CLEAN_NII_FILES = glob_files(GENERAL_FOLDER_NAME, "nii.gz")

    # Filter for the specific one we want
    CLEAN_NII_FILES = [file for file in CLEAN_NII_FILES if "clean_nii" in file]

    # Return the first one
    return CLEAN_NII_FILES[0]

# ----------------------------------------------------- COMMAND DEFINITIONS ----------------------------------------------------- #

# Define MRTrix General (mask) commands
def define_mrtrix_general_commands(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    DWI_FILES = ARGS[1]
    FOLDER_TYPE = ARGS[2]

    # Get the rest of the paths for the commands
    (INPUT_MIF_PATH, MASK_MIF_PATH, MASK_NII_PATH) = get_mrtrix_general_paths(REGION_ID, DWI_FILES, FOLDER_TYPE)

    # DWI brain mask and conversion mif -> nii command
    MASK_CMD = "dwi2mask {input}.mif {output}.mif".format(input=INPUT_MIF_PATH, output=MASK_MIF_PATH)
    MASK_NII_CMD = "mrconvert {input}.mif {output}.nii".format(input=MASK_MIF_PATH, output=MASK_NII_PATH)

    # Return the commands
    return (MASK_CMD, MASK_NII_CMD)

# Define MRtrix cleaning commands
def define_mrtrix_clean_commands(ARGS):
    
    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    DWI_FILES = ARGS[1]
    FOLDER_TYPE = ARGS[2]

    # Get the general paths
    (INPUT_MIF_PATH, MASK_MIF_PATH, MASK_NII_PATH) = get_mrtrix_general_paths(REGION_ID, DWI_FILES, FOLDER_TYPE)
    # Get the clean paths
    (DWI_B0_PATH, DWI_B0_NII, SKULL_STRIP_PATH, SKULL_STRIP_MIF_PATH, DWI_DENOISE_PATH, DWI_NOISE_PATH, 
    DWI_EDDY_PATH, DWI_CLEAN_MIF_PATH, DWI_CLEAN_MASK_PATH, DWI_CLEAN_MASK_NII_PATH, DWI_CLEAN_NII_PATH, 
    DWI_CLEAN_BVEC_PATH, DWI_CLEAN_BVAL_PATH) = get_mrtrix_clean_paths(REGION_ID, FOLDER_TYPE)
    
    # Get the input paths
    DWI_NEEDED = ["dwi_nii", "bval", "bvec"]
    DWI_NEEDED_PATHS = extract_from_input_list(DWI_FILES, DWI_NEEDED, "dwi")

    # Denoise command
    DWI_DENOISE_CMD = "dwidenoise {input}.mif {output}.mif -noise {noise}.mif".format(
        input=INPUT_MIF_PATH, output=DWI_DENOISE_PATH, noise=DWI_NOISE_PATH)
    # Bias correction command
    DWI_BIAS_CMD = "dwibiascorrect ants {input}.mif {output}.mif".format(input=DWI_DENOISE_PATH, output=DWI_CLEAN_MIF_PATH)
    # Convert to NII command
    DWI_CONVERT_CMD = "mrconvert {input}.mif {output}.nii.gz -export_grad_fsl {bvec_clean}.bvec {bval_clean}.bval".format(
        input=DWI_CLEAN_MIF_PATH, output=DWI_CLEAN_NII_PATH, bvec_clean=DWI_CLEAN_BVEC_PATH, bval_clean=DWI_CLEAN_BVAL_PATH)
    # Get the mask
    CLEAN_MASK_CMD = "dwi2mask {input}.mif {output}.mif".format(input=DWI_CLEAN_MIF_PATH, output=DWI_CLEAN_MASK_PATH)
    # Convert mask to NII command
    CLEAN_MASK_NII_CMD = "mrconvert {input}.mif {output}.nii.gz".format(input=DWI_CLEAN_MASK_PATH, output=DWI_CLEAN_MASK_NII_PATH)

    # Return the commands
    return (DWI_DENOISE_CMD, DWI_BIAS_CMD, DWI_CONVERT_CMD, CLEAN_MASK_CMD, CLEAN_MASK_NII_CMD)

# Define the template registration commands
def define_template_registration_commands(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]
    FOLDER_TYPE = ARGS[2]

    # Define what's needed for this registration and extract them from subject files
    ATLAS_NEEDED = ["stpt"]
    ATLAS_NEEDED_PATH = extract_from_input_list(ATLAS_STPT, ATLAS_NEEDED, "atlas_stpt")

    # Get the clean paths
    (DWI_B0_PATH, DWI_B0_NII, SKULL_STRIP_PATH, SKULL_STRIP_MIF_PATH, DWI_DENOISE_PATH, DWI_NOISE_PATH, 
    DWI_EDDY_PATH, DWI_CLEAN_MIF_PATH, DWI_CLEAN_MASK_PATH, DWI_CLEAN_MASK_NII_PATH, DWI_CLEAN_NII_PATH, 
    DWI_CLEAN_BVEC_PATH, DWI_CLEAN_BVAL_PATH) = get_mrtrix_clean_paths(REGION_ID, FOLDER_TYPE)
    # Get the registration paths
    (DWI_REG_FOLDER, DWI_MAP_MAT, DWI_CONVERT_INV, DWI_REG_PATH, 
     DWI_REG_MASK_PATH) = get_STPT_registration_paths(REGION_ID, FOLDER_TYPE)
    
    # Get the reference image
    REFERENCE_IMAGE = get_clean_reference(hpc=True)
    
    # Transformation and registration of the clean mif to the template space
    REGISTER_DWI_STPT_CMD = "flirt -in {stpt} -ref {dwi} -interp nearestneighbour -dof 6 -omat {transform_mat}.mat".format(
        stpt=REFERENCE_IMAGE, dwi=DWI_CLEAN_NII_PATH, transform_mat=DWI_MAP_MAT)
    TRANSFORMATION_DWI_STPT_CMD = "transformconvert {transform_mat}.mat {stpt} {dwi} flirt_import {output}.txt".format(
        transform_mat=DWI_MAP_MAT, stpt=REFERENCE_IMAGE, dwi=DWI_CLEAN_NII_PATH, output=DWI_CONVERT_INV)
    FINAL_DWI_TRANSFORM_CMD = "mrtransform {dwi}.mif -linear {transform}.txt -inverse {output}.mif".format(
        dwi=DWI_CLEAN_MIF_PATH, transform=DWI_CONVERT_INV, output=DWI_REG_PATH)
    # Create mask for the registered DWI
    FINAL_DWI_MASK_CMD = "dwi2mask {input}.mif {output}.mif".format(input=DWI_REG_PATH, output=DWI_REG_MASK_PATH)

    # Return the commands
    return (REGISTER_DWI_STPT_CMD, TRANSFORMATION_DWI_STPT_CMD, FINAL_DWI_TRANSFORM_CMD, FINAL_DWI_MASK_CMD)

# Define MRtrix FOD commands
def define_mrtrix_fod_commands(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    FOLDER_TYPE = ARGS[1]
    
    # Get the clean paths
    (DWI_B0_PATH, DWI_B0_NII, SKULL_STRIP_PATH, SKULL_STRIP_MIF_PATH, DWI_DENOISE_PATH, DWI_NOISE_PATH, 
    DWI_EDDY_PATH, DWI_CLEAN_MIF_PATH, DWI_CLEAN_MASK_PATH, DWI_CLEAN_MASK_NII_PATH, DWI_CLEAN_NII_PATH, 
    DWI_CLEAN_BVEC_PATH, DWI_CLEAN_BVAL_PATH) = get_mrtrix_clean_paths(REGION_ID, FOLDER_TYPE)
    # Get the registration paths
    (DWI_REG_FOLDER, DWI_MAP_MAT, DWI_CONVERT_INV, DWI_REG_PATH, 
     DWI_REG_MASK_PATH) = get_STPT_registration_paths(REGION_ID, FOLDER_TYPE)
    # Get the fod paths
    (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH,
    WM_FOD_PATH, GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, 
    CSF_FOD_NORM_PATH) = get_mrtrix_fod_paths(REGION_ID, FOLDER_TYPE)

    # Depending on if we're of folder type BMCR or BMA, we need to use different mifs and masks
    if FOLDER_TYPE == "BMCR" or FOLDER_TYPE == "BMA_EXVIVO":
        INPUT_MIF_PATH = DWI_CLEAN_MIF_PATH
        INPUT_MASK_PATH = DWI_CLEAN_MASK_PATH
    elif FOLDER_TYPE == "BMA_INVIVO":
        INPUT_MIF_PATH = DWI_REG_PATH
        INPUT_MASK_PATH = DWI_REG_MASK_PATH

    # Response estimation of WM, GM, CSF from DWI command
    RESPONSE_EST_CMD = "dwi2response dhollander {input}.mif -mask {mask}.mif {wm}.txt {gm}.txt {csf}.txt -voxels \
        {response_voxels}.mif".format(input=INPUT_MIF_PATH, mask=INPUT_MASK_PATH, wm=RESPONSE_WM_PATH, 
                                      gm=RESPONSE_GM_PATH, csf=RESPONSE_CSF_PATH, 
                                      response_voxels=RESPONSE_VOXEL_PATH)
    VIEW_RESPONSE_CMD = "mrview {input}.mif -overlay.load {response_voxels}.mif".format(input=INPUT_MIF_PATH,
                                                                                        response_voxels=RESPONSE_VOXEL_PATH)    
    # Spherical deconvolution to estimate fODs command
    MULTISHELL_CSD_CMD = "dwi2fod msmt_csd {input}.mif {wm}.txt {wmfod}.mif {gm}.txt {gmfod}.mif {csf}.txt \
        {csffod}.mif -mask {mask}.mif".format(
        input=INPUT_MIF_PATH, wm=RESPONSE_WM_PATH, wmfod=WM_FOD_PATH, gm=RESPONSE_GM_PATH, gmfod=GM_FOD_PATH,
        csf=RESPONSE_CSF_PATH, csffod=CSF_FOD_PATH, mask=INPUT_MASK_PATH)
    # Combining fODs into a VF command
    COMBINE_FODS_CMD = "mrconvert -coord 3 0 {wmfod}.mif - | mrcat {csffod}.mif {gmfod}.mif - {output}.mif".format(
        wmfod=WM_FOD_PATH, csffod=CSF_FOD_PATH, gmfod=GM_FOD_PATH, output=VF_FOD_PATH)
    VIEW_COMBINED_FODS_CMD = "mrview {vf}.mif -odf.load_sh {wmfod}.mif".format(vf=VF_FOD_PATH, wmfod=WM_FOD_PATH)
    # Normalizing fODs command
    NORMALIZE_FODS_CMD = "mtnormalise {wmfod}.mif {wmfod_norm}.mif {gmfod}.mif {gmfod_norm}.mif {csffod}.mif \
        {csffod_norm}.mif -mask {mask}.mif".format(
        wmfod=WM_FOD_PATH, wmfod_norm=WM_FOD_NORM_PATH, gmfod=GM_FOD_PATH, gmfod_norm=GM_FOD_NORM_PATH, csffod=CSF_FOD_PATH,
        csffod_norm=CSF_FOD_NORM_PATH, mask=INPUT_MASK_PATH)
    VIEW_NORMALIZED_FODS_CMD = "mrview {vf}.mif -odf.load_sh {wmfod_norm}.mif".format(vf=VF_FOD_PATH,
                                                                                      wmfod_norm=WM_FOD_NORM_PATH)
    
    # Return the commands
    return (RESPONSE_EST_CMD, VIEW_RESPONSE_CMD, MULTISHELL_CSD_CMD, COMBINE_FODS_CMD,
                VIEW_COMBINED_FODS_CMD, NORMALIZE_FODS_CMD, VIEW_NORMALIZED_FODS_CMD)

# Define the atlas Registration commands
def define_atlas_registration_commands(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]
    FOLDER_TYPE = ARGS[2]
    
    # Define what's needed for MRTRIX FOD and extract them from subject files
    ATLAS_NEEDED = ["atlas"]
    ATLAS_NEEDED_PATH = extract_from_input_list(ATLAS_STPT, ATLAS_NEEDED, "atlas_stpt")

    # Get the clean paths
    (DWI_B0_PATH, DWI_B0_NII, SKULL_STRIP_PATH, SKULL_STRIP_MIF_PATH, DWI_DENOISE_PATH, DWI_NOISE_PATH, 
    DWI_EDDY_PATH, DWI_CLEAN_MIF_PATH, DWI_CLEAN_MASK_PATH, DWI_CLEAN_MASK_NII_PATH, DWI_CLEAN_NII_PATH, 
    DWI_CLEAN_BVEC_PATH, DWI_CLEAN_BVAL_PATH) = get_mrtrix_clean_paths(REGION_ID, FOLDER_TYPE)
    # Get the registered DWI paths
    (DWI_REG_FOLDER, DWI_MAP_MAT, DWI_CONVERT_INV, DWI_REG_PATH, 
     DWI_REG_MASK_PATH) = get_STPT_registration_paths(REGION_ID, FOLDER_TYPE)
    # Get the registration paths
    (ATLAS_REG_FOLDER, DWI_B0_PATH, DWI_B0_NII, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, ATLAS_REG_PATH, 
    ATLAS_MIF_PATH) = get_atlas_registration_paths(REGION_ID, ATLAS_NEEDED_PATH, FOLDER_TYPE)

    # If we're of folder type BMCR, then everything's in STPT and we use the normal clean paths. If we're of folder type
    # BMA, then we need to use the STPT registered DWI paths
    if FOLDER_TYPE == "BMCR" or FOLDER_TYPE == "BMA_EXVIVO":
        INPUT_MIF_PATH = DWI_CLEAN_MIF_PATH
    elif FOLDER_TYPE == "BMA_INVIVO":
        INPUT_MIF_PATH = DWI_REG_PATH

    # Extracting mean B0 and transforming to NII command
    DWI_B0_CMD = "dwiextract {input}.mif - -bzero | mrmath - mean {output}.mif -axis 3".format(
        input=INPUT_MIF_PATH, output=DWI_B0_PATH)
    DWI_B0_NII_CMD = "mrconvert {input}.mif {output}".format(input=DWI_B0_PATH, output=DWI_B0_NII)
    # Transformation and registration of atlas to DWI space (to be used for connectome generation)
    REGISTER_ATLAS_DWI_CMD = "flirt -in {dwi} -ref {atlas} -interp nearestneighbour -dof 6 -omat {transform_mat}.mat".format(
        dwi=DWI_B0_NII, atlas=ATLAS_NEEDED_PATH["atlas"], transform_mat=ATLAS_DWI_MAP_MAT)
    TRANSFORMATION_ATLAS_DWI_CMD = "transformconvert {transform_mat}.mat {dwi} {atlas} flirt_import {output}.txt".format(
        transform_mat=ATLAS_DWI_MAP_MAT, dwi=DWI_B0_NII, atlas=ATLAS_NEEDED_PATH["atlas"], output=ATLAS_DWI_CONVERT_INV)
    ATLAS_MIF_CMD = "mrconvert {atlas} {output}.mif".format(atlas=ATLAS_NEEDED_PATH["atlas"], output=ATLAS_MIF_PATH)
    FINAL_ATLAS_TRANSFORM_CMD = "mrtransform {atlas}.mif -linear {transform}.txt -inverse {output}.mif".format(
        atlas=ATLAS_MIF_PATH, transform=ATLAS_DWI_CONVERT_INV, output=ATLAS_REG_PATH)
    
    # Return the commands
    return (DWI_B0_CMD, DWI_B0_NII_CMD, REGISTER_ATLAS_DWI_CMD, TRANSFORMATION_ATLAS_DWI_CMD,
            ATLAS_MIF_CMD, FINAL_ATLAS_TRANSFORM_CMD)

# Define MRtrix probabilistic tracking commands
def define_mrtrix_probtrack_commands(ARGS):
    
    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    DWI_FILES = ARGS[1]
    ATLAS_STPT = ARGS[2]
    FOLDER_TYPE = ARGS[3]

    # Define what's needed for MRTRIX FOD and extract them from subject files
    DWI_NEEDED = ["dwi_nii", "bval", "bvec"]
    DWI_NEEDED_PATHS = extract_from_input_list(DWI_FILES, DWI_NEEDED, "dwi")
    ATLAS_NEEDED = ["atlas"]
    ATLAS_NEEDED_PATH = extract_from_input_list(ATLAS_STPT, ATLAS_NEEDED, "atlas_stpt")

    # Get the clean paths
    (DWI_B0_PATH, DWI_B0_NII, SKULL_STRIP_PATH, SKULL_STRIP_MIF_PATH, DWI_DENOISE_PATH, DWI_NOISE_PATH, 
    DWI_EDDY_PATH, DWI_CLEAN_MIF_PATH, DWI_CLEAN_MASK_PATH, DWI_CLEAN_MASK_NII_PATH, DWI_CLEAN_NII_PATH, 
    DWI_CLEAN_BVEC_PATH, DWI_CLEAN_BVAL_PATH) = get_mrtrix_clean_paths(REGION_ID, FOLDER_TYPE)
    # Get the registered DWI paths
    (DWI_REG_FOLDER, DWI_MAP_MAT, DWI_CONVERT_INV, DWI_REG_PATH, 
     DWI_REG_MASK_PATH) = get_STPT_registration_paths(REGION_ID, FOLDER_TYPE)
    # Get the fod paths
    (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH,
    WM_FOD_PATH, GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, 
    CSF_FOD_NORM_PATH) = get_mrtrix_fod_paths(REGION_ID, FOLDER_TYPE)
    # Get the registration paths
    (ATLAS_REG_FOLDER, DWI_B0_PATH, DWI_B0_NII, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, ATLAS_REG_PATH, 
    ATLAS_MIF_PATH) = get_atlas_registration_paths(REGION_ID, ATLAS_NEEDED_PATH, FOLDER_TYPE)
    # Get the probabilistic tracking paths
    (GM_WM_SEED_PATH, TRACT_TCK_PATH) = get_mrtrix_probtrack_paths(REGION_ID, FOLDER_TYPE)

    # Depending on if we're of folder type BMCR or BMA, we need to use different masks
    if FOLDER_TYPE == "BMCR" or FOLDER_TYPE == "BMA_EXVIVO":
        INPUT_MASK_PATH = DWI_CLEAN_MASK_PATH
    elif FOLDER_TYPE == "BMA_INVIVO":
        INPUT_MASK_PATH = DWI_REG_MASK_PATH
    
    # Probabilistic tractography command
    PROB_TRACT_CMD = "tckgen {wmfod_norm}.mif {output}.tck -algorithm iFOD2 -seed_image {mask}.nii.gz -mask {mask}.nii.gz \
        -angle {opt_angle} -minlength {opt_minlength} -cutoff {opt_cutoff} \
        -fslgrad {bvec} {bval} -select 300000 -force".format(wmfod_norm=WM_FOD_NORM_PATH, output=TRACT_TCK_PATH, 
                                                             mask=INPUT_MASK_PATH,
                                                             opt_angle=32.2, opt_cutoff=0.05, opt_minlength=4.8,
                                                             bvec=DWI_NEEDED_PATHS["bvec"], bval=DWI_NEEDED_PATHS["bval"])
    
    # Return the commands
    return (PROB_TRACT_CMD)

# Define MRtrix connectome commands
def define_mrtrix_connectome_commands(ARGS):
    
    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]
    FOLDER_TYPE = ARGS[2]

    # Define what's needed for MRTRIX FOD and extract them from subject files
    ATLAS_NEEDED = ["atlas"]
    ATLAS_NEEDED_PATH = extract_from_input_list(ATLAS_STPT, ATLAS_NEEDED, "atlas_stpt")

    # Get the registration paths
    (ATLAS_REG_FOLDER, DWI_B0_PATH, DWI_B0_NII, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, ATLAS_REG_PATH, 
    ATLAS_MIF_PATH) = get_atlas_registration_paths(REGION_ID, ATLAS_NEEDED_PATH, FOLDER_TYPE)
    # Get the probabilistic tracking paths
    (GM_WM_SEED_PATH, TRACT_TCK_PATH) = get_mrtrix_probtrack_paths(REGION_ID, FOLDER_TYPE)
    # Get the connectivity paths
    (CONNECTIVITY_PROB_PATH) = get_mrtrix_connectome_paths(REGION_ID, FOLDER_TYPE)
    
    # Connectivity matrix command
    CONNECTIVITY_PROB_CMD = "tck2connectome {input}.tck {atlas}.mif {output}.csv -zero_diagonal -symmetric \
        -assignment_all_voxels -force".format(input=TRACT_TCK_PATH, output=CONNECTIVITY_PROB_PATH, atlas=ATLAS_REG_PATH)
    
    # Return the commands
    return (CONNECTIVITY_PROB_CMD)

# Define MRTRIX commands
def pre_tractography_commands(ARGS):
    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    DWI_FILES = ARGS[1]
    ATLAS_STPT = ARGS[2]
    FOLDER_TYPE = ARGS[3]

    # Define the general commands
    GENERAL_ARGS = [REGION_ID, DWI_FILES, FOLDER_TYPE]
    (MASK_CMD, MASK_NII_CMD) = define_mrtrix_general_commands(GENERAL_ARGS)

    # Define the cleaning commands
    CLEAN_ARGS = [REGION_ID, DWI_FILES, FOLDER_TYPE]
    (DWI_DENOISE_CMD, DWI_BIAS_CMD, DWI_CONVERT_CMD, CLEAN_MASK_CMD, 
     CLEAN_MASK_NII_CMD) = define_mrtrix_clean_commands(CLEAN_ARGS)

    # Define the registration commands, depending on if we're of folder type BMCR or BMA
    if FOLDER_TYPE == "BMA_INVIVO":
        # Need to register to STPT template space first
        REG_ARGS = [REGION_ID, ATLAS_STPT, FOLDER_TYPE]
        (REGISTER_DWI_STPT_CMD, TRANSFORMATION_DWI_STPT_CMD, FINAL_DWI_TRANSFORM_CMD,
         FINAL_DWI_MASK_CMD) = define_template_registration_commands(REG_ARGS)

    # Define the FOD commands
    FOD_ARGS = [REGION_ID, FOLDER_TYPE]
    (RESPONSE_EST_CMD, VIEW_RESPONSE_CMD, MULTISHELL_CSD_CMD, COMBINE_FODS_CMD,
    VIEW_COMBINED_FODS_CMD, NORMALIZE_FODS_CMD, VIEW_NORMALIZED_FODS_CMD) = define_mrtrix_fod_commands(FOD_ARGS)
    
    REG_ARGS = [REGION_ID, ATLAS_STPT, FOLDER_TYPE]
    (DWI_B0_CMD_ATLAS, DWI_B0_NII_CMD_ATLAS, REGISTER_ATLAS_DWI_CMD, TRANSFORMATION_ATLAS_DWI_CMD,
    ATLAS_MIF_CMD, FINAL_ATLAS_TRANSFORM_CMD) = define_atlas_registration_commands(REG_ARGS)

    # Define the probabilistic tractography commands
    PROB_ARGS = [REGION_ID, DWI_FILES, ATLAS_STPT, FOLDER_TYPE]
    (PROB_TRACT_CMD) = define_mrtrix_probtrack_commands(PROB_ARGS)

    # Define the connectome commands
    CONNECTOME_ARGS = [REGION_ID, ATLAS_STPT, FOLDER_TYPE]
    (CONNECTIVITY_PROB_CMD) = define_mrtrix_connectome_commands(CONNECTOME_ARGS)

    # Get the checkpoints for what has and hasn't been done yet
    CHECKPOINT_ARGS = [REGION_ID, DWI_FILES, ATLAS_STPT, FOLDER_TYPE]
    (MRTRIX_GENERAL, MRTRIX_DWI_REGISTRATION, MRTRIX_RESPONSE, MRTRIX_FOD, MRTRIX_FOD_NORM, 
    MRTRIX_ATLAS_REGISTRATION, MRTRIX_PROBTRACK, MRTRIX_CONNECTOME) = check_all_mrtrix_missing_files(CHECKPOINT_ARGS)

    # Print the checkpoints
    print("--- MRtrix General: {}".format(MRTRIX_GENERAL))
    print("--- MRtrix Response: {}".format(MRTRIX_RESPONSE))
    print("--- MRtrix FOD: {}".format(MRTRIX_FOD))
    print("--- MRtrix FOD Norm: {}".format(MRTRIX_FOD_NORM))
    print("--- MRtrix DWI Registration: {}".format(MRTRIX_DWI_REGISTRATION))
    print("--- MRtrix Atlas Registration: {}".format(MRTRIX_ATLAS_REGISTRATION))
    print("--- MRtrix Probtrack: {}".format(MRTRIX_PROBTRACK))
    print("--- MRtrix Connectome: {}".format(MRTRIX_CONNECTOME))

    # Define the commands array, depending on what's been done before
    MRTRIX_COMMANDS = []
    if MRTRIX_GENERAL:
        # First do the general stuff
        MRTRIX_COMMANDS.extend([
                                (MASK_CMD, "Create DWI brain mask"),
                                (MASK_NII_CMD, "Convert DWI brain mask mif -> nii"), 
                            ])
        # Now we do denoising, bias correction and conversion to NII
        MRTRIX_COMMANDS.extend([
                            (DWI_DENOISE_CMD, "Denoise DWI"),
                            (DWI_BIAS_CMD, "Bias correct DWI"),
                            (DWI_CONVERT_CMD, "Convert DWI mif -> nii"),
                            (CLEAN_MASK_CMD, "Create DWI brain mask for cleaned DWI"),
                            (CLEAN_MASK_NII_CMD, "Convert DWI brain mask mif -> nii for cleaned DWI"),
                        ])
    if MRTRIX_DWI_REGISTRATION and FOLDER_TYPE == "BMA_INVIVO":
        MRTRIX_COMMANDS.extend([
                                (REGISTER_DWI_STPT_CMD, "Begin registering DWI to STPT template space"),
                                (TRANSFORMATION_DWI_STPT_CMD, "Initial transformation of DWI to STPT template space"),
                                (FINAL_DWI_TRANSFORM_CMD, "Final transformation of DWI to STPT template space"),
                                (FINAL_DWI_MASK_CMD, "Create DWI brain mask for registered DWI"),
                            ])
    if MRTRIX_RESPONSE:
        MRTRIX_COMMANDS.extend([
                                (RESPONSE_EST_CMD, "Estimate response function of WM, GM, CSF from DWI"),
                            ])
    if MRTRIX_FOD:
        MRTRIX_COMMANDS.extend([
                                (MULTISHELL_CSD_CMD, "Spherical deconvolution to estimate fODs"),
                                (COMBINE_FODS_CMD, "Combining fODs into a VF"),
                            ])
    if MRTRIX_FOD_NORM:
        MRTRIX_COMMANDS.extend([
                                (NORMALIZE_FODS_CMD, "Normalizing fODs"),
                            ])
    if MRTRIX_ATLAS_REGISTRATION:
        MRTRIX_COMMANDS.extend([
                                (DWI_B0_CMD_ATLAS, "Extracting mean B0 and transforming to NII"), (DWI_B0_NII_CMD_ATLAS, "DWI B0 mif -> NII"),
                                (REGISTER_ATLAS_DWI_CMD, "Begin registering atlas to DWI space"),
                                (TRANSFORMATION_ATLAS_DWI_CMD, "Initial transformation of atlas to DWI space"),
                                (ATLAS_MIF_CMD, "Convert atlas nii -> mif"), (FINAL_ATLAS_TRANSFORM_CMD, "Final transformation of atlas to DWI space"),
                        ])
    if MRTRIX_PROBTRACK:
        MRTRIX_COMMANDS.extend([
                                (PROB_TRACT_CMD, "Probabilistic tractography"),
                            ])
    if MRTRIX_CONNECTOME:
        MRTRIX_COMMANDS.extend([
                                (CONNECTIVITY_PROB_CMD, "Creating connectivity matrix - probabilistic"),
                            ])
    
    # Return the commands array
    return MRTRIX_COMMANDS
