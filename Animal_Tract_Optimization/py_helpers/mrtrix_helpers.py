from .general_helpers import *
import os

# ----------------------------------------------------- PATH DEFINITIONS ----------------------------------------------------- #

# Define MRTRIX folder paths
def main_mrtrix_folder_paths(REGION_ID):
    
    # Get the general paths
    (BMINDS_DATA_FOLDER, BMINDS_OUTPUTS_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_INJECTIONS_FOLDER, 
        BMINDS_ATLAS_STPT_FOLDER, BMINDS_ZIPPED_DWI_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER, 
            MAIN_MRTRIX_FOLDER) = get_main_paths(hpc=False)

    # Create the region's main folder - don't wipe as it'll be invoked more than once
    REGION_MRTRIX_PATH = os.path.join(MAIN_MRTRIX_FOLDER, REGION_ID)
    check_output_folders(REGION_MRTRIX_PATH, "REGION MRTRIX PATH", wipe=False)
    
    # Because there's a lot of commands, we define extra folder paths here. DON'T WIPE as it'll be used by other functions
    GENERAL_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "general")
    RESPONSE_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "response")
    FOD_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "fod")
    FOD_NORM_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "fod_norm")
    # T1_REG_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "t1_and_Fivett_reg")
    ATLAS_REG_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "atlas_reg")
    PROB_TRACKING_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "prob_tracking")
    # GLOBAL_TRACKING_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "global_tracking")
    CONNECTIVITY_FOLDER_NAME = os.path.join(REGION_MRTRIX_PATH, "connectivity")

    # Check the output folders
    check_output_folders(GENERAL_FOLDER_NAME, "MRtrix general folder", wipe=False)
    check_output_folders(RESPONSE_FOLDER_NAME, "MRtrix response folder", wipe=False)
    check_output_folders(FOD_FOLDER_NAME, "MRtrix fod folder", wipe=False)
    check_output_folders(FOD_NORM_FOLDER_NAME, "MRtrix fod norm folder", wipe=False)
    # check_output_folders(T1_REG_FOLDER_NAME, "MRtrix t1 and Fivett reg folder", wipe=False)
    check_output_folders(ATLAS_REG_FOLDER_NAME, "MRtrix atlas reg folder", wipe=False)
    check_output_folders(PROB_TRACKING_FOLDER_NAME, "MRtrix probabilistic tracking folder", wipe=False)
    # check_output_folders(GLOBAL_TRACKING_FOLDER_NAME, "MRtrix global tracking folder", wipe=False)
    check_output_folders(CONNECTIVITY_FOLDER_NAME, "MRtrix connectivity folder", wipe=False)

    # Return the paths
    return (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
            FOD_NORM_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, CONNECTIVITY_FOLDER_NAME)

# Define MRtrix general paths
def get_mrtrix_general_paths(REGION_ID):
    
    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
        CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(REGION_ID)

    # DWI nii -> mif + mask filepaths
    INPUT_MIF_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_mif".format(REGION_ID))
    MASK_MIF_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_mask_mif".format(REGION_ID))
    MASK_NII_PATH = os.path.join(GENERAL_FOLDER_NAME, "{}_mask_nii".format(REGION_ID))

    # Return the paths
    return (INPUT_MIF_PATH, MASK_MIF_PATH, MASK_NII_PATH)
    
# Define MRTrix FOD paths
def get_mrtrix_fod_paths(REGION_ID):
    
    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
        CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(REGION_ID)

    # Finding the response function paths
    RESPONSE_WM_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_wm".format(REGION_ID))
    RESPONSE_GM_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_gm".format(REGION_ID))
    RESPONSE_CSF_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_csf".format(REGION_ID))
    RESPONSE_VOXEL_PATH = os.path.join(RESPONSE_FOLDER_NAME, "{}_response_voxels".format(REGION_ID))
    # Finding the fiber orientation distributions (fODs)
    WM_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_wmfod".format(REGION_ID))
    GM_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_gmfod".format(REGION_ID))
    CSF_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_csffod".format(REGION_ID))
    VF_FOD_PATH = os.path.join(FOD_FOLDER_NAME, "{}_vf".format(REGION_ID))
    # Normalizing the fiber orientation distributions (fODs)
    WM_FOD_NORM_PATH = os.path.join(FOD_NORM_FOLDER_NAME, "{}_wmfod_norm".format(REGION_ID))
    GM_FOD_NORM_PATH = os.path.join(FOD_NORM_FOLDER_NAME, "{}_gmfod_norm".format(REGION_ID))
    CSF_FOD_NORM_PATH = os.path.join(FOD_NORM_FOLDER_NAME, "{}_csffod_norm".format(REGION_ID))
    
    # Return the paths
    return (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH,
                WM_FOD_PATH, GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, CSF_FOD_NORM_PATH)

# Define MRtrix Registration paths
def get_mrtrix_registration_paths(REGION_ID, ATLAS):
    
    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
            CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(REGION_ID)

    # Convert atlas nii to mif, then register/map it to the B0 space of DWI
    DWI_B0_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_b0".format(REGION_ID))
    DWI_B0_NII = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_b0.nii.gz".format(REGION_ID))

    ATLAS_DWI_MAP_MAT = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlas2dwi_fsl".format(REGION_ID))
    ATLAS_DWI_CONVERT_INV = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlas2dwi_mrtrix".format(REGION_ID))
    ATLAS_REG_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlasreg".format(REGION_ID))
    # Getting the name of the atlas without .nii.gz
    ATLAS_NAME = ATLAS["atlas"].split("/")[-1].split(".")[0]
    ATLAS_MIF_PATH = os.path.join(ATLAS_REG_FOLDER_NAME, "{}_atlas_mif".format(ATLAS_NAME))

    # Return the paths
    return (DWI_B0_PATH, DWI_B0_NII, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, ATLAS_REG_PATH, ATLAS_MIF_PATH)

# Define MRtrix probabilistic tracking paths
def get_mrtrix_probtrack_paths(REGION_ID):

    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
            CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(REGION_ID)

    # Mask for streamline seeding paths and probabilistic tractography path
    GM_WM_SEED_PATH = os.path.join(PROB_TRACKING_FOLDER_NAME, "{}_gmwmseed".format(REGION_ID))
    TRACT_TCK_PATH = os.path.join(PROB_TRACKING_FOLDER_NAME, "{}_tract".format(REGION_ID))

    # Return the paths
    return (GM_WM_SEED_PATH, TRACT_TCK_PATH)

# Define MRtrix connectome paths
def get_mrtrix_connectome_paths(REGION_ID):

    # Get the folder names
    (REGION_MRTRIX_PATH, GENERAL_FOLDER_NAME, RESPONSE_FOLDER_NAME, FOD_FOLDER_NAME, 
        FOD_NORM_FOLDER_NAME, ATLAS_REG_FOLDER_NAME, PROB_TRACKING_FOLDER_NAME, 
            CONNECTIVITY_FOLDER_NAME) = main_mrtrix_folder_paths(REGION_ID)

    # Connectivity matrix path
    CONNECTIVITY_PROB_PATH = os.path.join(CONNECTIVITY_FOLDER_NAME, "{}_prob_connectivity".format(REGION_ID))

    # Return the paths
    return (CONNECTIVITY_PROB_PATH)

# ----------------------------------------------------- COMMAND DEFINITIONS ----------------------------------------------------- #

# Define MRtrix FOD commands
def define_mrtrix_fod_commands(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    DWI_FILES = ARGS[1]
    
    # Define what's needed for MRTRIX FOD and extract them from subject files
    DWI_NEEDED = ["dwi", "bval", "bvec"]
    DWI_NEEDED_PATHS = extract_from_input_list(DWI_FILES, DWI_NEEDED, "dwi")

    # Get the rest of the paths for the commands
    (INPUT_MIF_PATH, MASK_MIF_PATH, MASK_NII_PATH) = get_mrtrix_general_paths(REGION_ID)
    # Get the fod paths
    (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH,
        WM_FOD_PATH, GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, 
            CSF_FOD_NORM_PATH) = get_mrtrix_fod_paths(REGION_ID)


    # DWI nii -> mif command
    MIF_CMD = "mrconvert {input_nii} -fslgrad {bvec} {bval} {output}.mif".format(input_nii=DWI_NEEDED_PATHS["dwi"], 
                                                                                bvec=DWI_NEEDED_PATHS["bvec"], 
                                                                                bval=DWI_NEEDED_PATHS["bval"], 
                                                                                output=INPUT_MIF_PATH)
    # DWI brain mask and conversion mif -> nii command
    MASK_CMD = "dwi2mask {input}.mif {output}.mif".format(input=INPUT_MIF_PATH, output=MASK_MIF_PATH)
    MASK_NII_CMD = "mrconvert {input}.mif {output}.nii".format(input=MASK_MIF_PATH, output=MASK_NII_PATH)
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
    return (MIF_CMD, MASK_CMD, MASK_NII_CMD, RESPONSE_EST_CMD, VIEW_RESPONSE_CMD, MULTISHELL_CSD_CMD, COMBINE_FODS_CMD,
                VIEW_COMBINED_FODS_CMD, NORMALIZE_FODS_CMD, VIEW_NORMALIZED_FODS_CMD)

# Define MRtrix Registration commands
def define_mrtrix_registration_commands(ARGS):

    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]
    
    # Define what's needed for MRTRIX FOD and extract them from subject files
    ATLAS_NEEDED = ["atlas"]
    ATLAS_NEEDED_PATH = extract_from_input_list(ATLAS_STPT, ATLAS_NEEDED, "atlas_stpt")

    # Get the rest of the paths for the commands
    (INPUT_MIF_PATH, MASK_MIF_PATH, MASK_NII_PATH) = get_mrtrix_general_paths(REGION_ID)
    # Get the registration paths
    (DWI_B0_PATH, DWI_B0_NII, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, 
     ATLAS_REG_PATH, ATLAS_MIF_PATH) = get_mrtrix_registration_paths(REGION_ID, ATLAS_NEEDED_PATH)

    # Extracting mean B0 and transforming to NII command
    DWI_B0_CMD = "dwiextract {input}.mif - -bzero | mrmath - mean {output}.mif -axis 3".format(
        input=INPUT_MIF_PATH, output=DWI_B0_PATH)
    DWI_B0_NII_CMD = "mrconvert {input}.mif {output}".format(input=DWI_B0_PATH, output=DWI_B0_NII)
    # Transformation and registration of atlas to DWI space (to be used for connectome generation)
    REGISTER_ATLAS_DWI_CMD = "flirt -in {dwi} -ref {atlas} -interp nearestneighbour -dof 6 -omat {transform_mat}.mat".format(
        dwi=DWI_B0_NII, atlas=ATLAS_NEEDED_PATH, transform_mat=ATLAS_DWI_MAP_MAT)
    TRANSFORMATION_ATLAS_DWI_CMD = "transformconvert {transform_mat}.mat {dwi} {atlas} flirt_import {output}.txt".format(
        transform_mat=ATLAS_DWI_MAP_MAT, dwi=DWI_B0_NII, atlas=ATLAS_NEEDED_PATH, output=ATLAS_DWI_CONVERT_INV)
    ATLAS_MIF_CMD = "mrconvert {atlas} {output}.mif".format(atlas=ATLAS_NEEDED_PATH, output=ATLAS_MIF_PATH)
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

    # Define what's needed for MRTRIX FOD and extract them from subject files
    DWI_NEEDED = ["dwi", "bval", "bvec"]
    DWI_NEEDED_PATHS = extract_from_input_list(DWI_FILES, DWI_NEEDED, "dwi")
    ATLAS_NEEDED = ["atlas"]
    ATLAS_NEEDED_PATH = extract_from_input_list(ATLAS_STPT, ATLAS_NEEDED, "atlas_stpt")

    # Get the general paths
    (INPUT_MIF_PATH, MASK_MIF_PATH, MASK_NII_PATH) = get_mrtrix_general_paths(REGION_ID)
    # Get the fod paths
    (RESPONSE_WM_PATH, RESPONSE_GM_PATH, RESPONSE_CSF_PATH, RESPONSE_VOXEL_PATH,
        WM_FOD_PATH, GM_FOD_PATH, CSF_FOD_PATH, VF_FOD_PATH, WM_FOD_NORM_PATH, GM_FOD_NORM_PATH, 
            CSF_FOD_NORM_PATH) = get_mrtrix_fod_paths(REGION_ID)
    # Get the registration paths
    (DWI_B0_PATH, DWI_B0_NII, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, 
     ATLAS_REG_PATH, ATLAS_MIF_PATH) = get_mrtrix_registration_paths(REGION_ID, ATLAS_NEEDED_PATH)
    # Get the probabilistic tracking paths
    (GM_WM_SEED_PATH, TRACT_TCK_PATH) = get_mrtrix_probtrack_paths(REGION_ID)
    
    # Probabilistic tractography command
    PROB_TRACT_CMD = "tckgen {wmfod_norm}.mif {output}.tck -algorithm iFOD2 -seed_image {mask}.nii -mask {mask}.nii \
        -angle {opt_angle} -minlength {opt_minlength} -cutoff {opt_cutoff} \
        -grad {bvec} {bval} -select 300000 -force".format(wmfod_norm=WM_FOD_NORM_PATH, output=TRACT_TCK_PATH, mask=MASK_NII_PATH,
                                                            opt_angle=32.2, opt_cutoff=0.05, opt_minlength=4.8,
                                                            bvec=DWI_NEEDED_PATHS["bvec"], bval=DWI_NEEDED_PATHS["bval"])
    
    # Return the commands
    return (PROB_TRACT_CMD)

# Define MRtrix connectome commands
def define_mrtrix_connectome_commands(ARGS):
    
    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    ATLAS_STPT = ARGS[1]

    # Define what's needed for MRTRIX FOD and extract them from subject files
    ATLAS_NEEDED = ["atlas"]
    ATLAS_NEEDED_PATH = extract_from_input_list(ATLAS_STPT, ATLAS_NEEDED, "atlas_stpt")


    # Get the registration paths
    (DWI_B0_PATH, DWI_B0_NII, ATLAS_DWI_MAP_MAT, ATLAS_DWI_CONVERT_INV, 
     ATLAS_REG_PATH, ATLAS_MIF_PATH) = get_mrtrix_registration_paths(REGION_ID, ATLAS_NEEDED_PATH)
    # Get the probabilistic tracking paths
    (GM_WM_SEED_PATH, TRACT_TCK_PATH) = get_mrtrix_probtrack_paths(REGION_ID)
    # Get the connectivity paths
    (CONNECTIVITY_PROB_PATH) = get_mrtrix_connectome_paths(REGION_ID)
    
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

    # Define the FOD commands
    FOD_ARGS = [REGION_ID, DWI_FILES]
    (MIF_CMD, MASK_CMD, MASK_NII_CMD, RESPONSE_EST_CMD, VIEW_RESPONSE_CMD, MULTISHELL_CSD_CMD, COMBINE_FODS_CMD,
        VIEW_COMBINED_FODS_CMD, NORMALIZE_FODS_CMD, VIEW_NORMALIZED_FODS_CMD) = define_mrtrix_fod_commands(FOD_ARGS)
    
    # Define the registration commands
    REG_ARGS = [REGION_ID, ATLAS_STPT]
    (DWI_B0_CMD, DWI_B0_NII_CMD, REGISTER_ATLAS_DWI_CMD, TRANSFORMATION_ATLAS_DWI_CMD,
        ATLAS_MIF_CMD, FINAL_ATLAS_TRANSFORM_CMD) = define_mrtrix_registration_commands(REG_ARGS)

    # Define the probabilistic tractography commands
    PROB_ARGS = [REGION_ID, DWI_FILES, ATLAS_STPT]
    (PROB_TRACT_CMD) = define_mrtrix_probtrack_commands(PROB_ARGS)

    # Define the connectome commands
    CONNECTOME_ARGS = [REGION_ID, ATLAS_STPT]
    (CONNECTIVITY_PROB_CMD) = define_mrtrix_connectome_commands(CONNECTOME_ARGS)

    # Create commands array
    MRTRIX_COMMANDS = [(MIF_CMD, "Convert DWI nii -> mif"), (MASK_CMD, "Create DWI brain mask"),
                        (MASK_NII_CMD, "Convert DWI brain mask mif -> nii"), 
                        (RESPONSE_EST_CMD, "Estimate response function of WM, GM, CSF from DWI"),
                        (MULTISHELL_CSD_CMD, "Spherical deconvolution to estimate fODs"),
                        (COMBINE_FODS_CMD, "Combining fODs into a VF"),
                        (NORMALIZE_FODS_CMD, "Normalizing fODs"),
                        (DWI_B0_CMD, "Extracting mean B0 and transforming to NII"), (DWI_B0_NII_CMD, "DWI B0 mif -> NII"),
                        (REGISTER_ATLAS_DWI_CMD, "Begin registering atlas to DWI space"),
                        (TRANSFORMATION_ATLAS_DWI_CMD, "Initial transformation of atlas to DWI space"),
                        (ATLAS_MIF_CMD, "Convert atlas nii -> mif"), (FINAL_ATLAS_TRANSFORM_CMD, "Final transformation of atlas to DWI space"),
                        (PROB_TRACT_CMD, "Probabilistic tractography"), (CONNECTIVITY_PROB_CMD, "Creating connectivity matrix - probabilistic")]
    
    # Return the commands array
    return MRTRIX_COMMANDS

# Define command to make the injection matrix
def def_injection_matrix_command(ARGS):
    # Extract arguments needed to define paths
    REGION_ID = ARGS[0]
    DWI_FILES = ARGS[1]
    STREAMLINE_FILES = ARGS[2]
    INJECTION_FILES = ARGS[3]
    ATLAS_STPT = ARGS[2]


    # Get the probabilistic tractography results path
    (GMWM_SEED_PATH, TRACT_TCK_PATH) = get_mrtrix_probtrack_paths(REGION_ID)


# # Define MRTRIX commands
# def probabilistic_tractography(ARGS):
#     # Extract arguments needed to define paths
#     REGION_ID = ARGS[0]
#     DWI_FILES = ARGS[1]
#     ATLAS_STPT = ARGS[2]

#     # Define the FOD commands
#     FOD_ARGS = [REGION_ID, DWI_FILES]
#     (MIF_CMD, MASK_CMD, MASK_NII_CMD, RESPONSE_EST_CMD, VIEW_RESPONSE_CMD, MULTISHELL_CSD_CMD, COMBINE_FODS_CMD,
#         VIEW_COMBINED_FODS_CMD, NORMALIZE_FODS_CMD, VIEW_NORMALIZED_FODS_CMD) = define_mrtrix_fod_commands(FOD_ARGS)
    
#     # Define the registration commands
#     REG_ARGS = [REGION_ID, ATLAS_STPT]
#     (DWI_B0_CMD, DWI_B0_NII_CMD, REGISTER_ATLAS_DWI_CMD, TRANSFORMATION_ATLAS_DWI_CMD,
#         ATLAS_MIF_CMD, FINAL_ATLAS_TRANSFORM_CMD) = define_mrtrix_registration_commands(REG_ARGS)
    
#     # Define the probabilistic tracking commands
#     PROB_ARGS = [REGION_ID, DWI_FILES, ATLAS_STPT]
#     (PROB_TRACT_CMD) = define_mrtrix_probtrack_commands(PROB_ARGS)

#     # Define the connectome commands
#     CONNECTOME_ARGS = [REGION_ID, ATLAS_STPT]
#     (CONNECTIVITY_PROB_CMD) = define_mrtrix_connectome_commands(CONNECTOME_ARGS)


#     # Create commands array
#     MRTRIX_COMMANDS = [(MIF_CMD, "Convert DWI nii -> mif"), (MASK_CMD, "Create DWI brain mask"),
#                         (MASK_NII_CMD, "Convert DWI brain mask mif -> nii"), 
#                         (RESPONSE_EST_CMD, "Estimate response function of WM, GM, CSF from DWI"),
#                         (MULTISHELL_CSD_CMD, "Spherical deconvolution to estimate fODs"),
#                         (COMBINE_FODS_CMD, "Combining fODs into a VF"),
#                         (NORMALIZE_FODS_CMD, "Normalizing fODs"),
#                         (DWI_B0_CMD, "Extracting mean B0 and transforming to NII"), (DWI_B0_NII_CMD, "DWI B0 mif -> NII"),
#                         (REGISTER_ATLAS_DWI_CMD, "Begin registering atlas to DWI space"),
#                         (TRANSFORMATION_ATLAS_DWI_CMD, "Initial transformation of atlas to DWI space"),
#                         (ATLAS_MIF_CMD, "Convert atlas nii -> mif"), (FINAL_ATLAS_TRANSFORM_CMD, "Final transformation of atlas to DWI space"),
#                         (PROB_TRACT_CMD, "Probabilistic tractography"), 
#                         (CONNECTIVITY_PROB_CMD, "Creating connectivity matrix - probabilistic")]

#     # Return the commands array
#     return MRTRIX_COMMANDS