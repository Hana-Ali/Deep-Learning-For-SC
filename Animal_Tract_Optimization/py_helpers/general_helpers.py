import os
import glob
import sys
import shutil

# Get the main paths
def get_main_paths(hpc):
    # Depending on whether we're in HPC or not, paths change
    if hpc == True:
        BMINDS_DATA_FOLDER = "/rds/general/user/hsa22/ephemeral/Brain_MINDS"
        BMINDS_OUTPUTS_DMRI_FOLDER = os.path.realpath(os.path.join(BMINDS_DATA_FOLDER, "processed_dMRI"))
        BMINDS_OUTPUTS_INJECTIONS_FOLDER = os.path.realpath(os.path.join(BMINDS_DATA_FOLDER, "processed_injections"))        
        BMINDS_CORE_FOLDER = os.path.realpath(os.path.join(BMINDS_DATA_FOLDER, "BMCR_core_data"))
        BMINDS_DWI_FOLDER = os.path.realpath(os.path.join(BMINDS_CORE_FOLDER, "dMRI_raw"))
        BMINDS_METADATA_FOLDER = os.path.realpath(os.path.join(BMINDS_CORE_FOLDER, "meta_data"))
        BMINDS_TEMPLATES_FOLDER = os.path.realpath(os.path.join(BMINDS_DATA_FOLDER, "BMCR_STPT_template"))
        BMINDS_ATLAS_FOLDER = os.path.realpath(os.path.join(BMINDS_TEMPLATES_FOLDER, "Atlases"))
        BMINDS_STPT_TEMPLATE_FOLDER = os.path.realpath(os.path.join(BMINDS_TEMPLATES_FOLDER, "STPT_population_average"))
        BMINDS_TRANSFORMS_FOLDER = os.path.realpath(os.path.join(BMINDS_TEMPLATES_FOLDER, "ANTS_transforms"))
        BMINDS_INJECTIONS_FOLDER = os.path.realpath(os.path.join(BMINDS_CORE_FOLDER, "processed_tracer_data"))
        BMINDS_UNZIPPED_DWI_FOLDER = os.path.realpath(os.path.join(BMINDS_OUTPUTS_DMRI_FOLDER, "dMRI_unzipped"))

    else:
        # Define paths based on whether we're Windows or Linux
        if os.name == "nt":
            ALL_DATA_FOLDER = os.path.realpath(r"C:\\tractography\\data")
            
            SUBJECTS_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "subjects"))
            TRACTOGRAPHY_OUTPUT_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "dsi_outputs"))
            NIPYPE_OUTPUT_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "nipype_outputs"))
            DWI_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "dwi"))
            T1_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "t1"))
            FMRI_MAIN_FOLDER = os.path.realpath(os.path.join(SUBJECTS_FOLDER, "fmri"))

            DSI_COMMAND = "dsi_studio"

            ATLAS_FOLDER = os.path.realpath(os.path.join(ALL_DATA_FOLDER, "atlas"))

        else:
            BMINDS_DATA_FOLDER = os.path.realpath(os.path.join(os.getcwd(), "..", "Animal_Data", "Brain-MINDS"))
            BMINDS_OUTPUTS_DMRI_FOLDER = os.path.realpath(os.path.join(BMINDS_DATA_FOLDER, "processed_dMRI"))
            BMINDS_OUTPUTS_INJECTIONS_FOLDER = os.path.realpath(os.path.join(BMINDS_DATA_FOLDER, "processed_injections"))
            BMINDS_CORE_FOLDER = os.path.realpath(os.path.join(BMINDS_DATA_FOLDER, "BMCR_core_data"))
            BMINDS_DWI_FOLDER = os.path.realpath(os.path.join(BMINDS_CORE_FOLDER, "dMRI_raw"))
            BMINDS_METADATA_FOLDER = os.path.realpath(os.path.join(BMINDS_CORE_FOLDER, "meta_data"))
            BMINDS_TEMPLATES_FOLDER = os.path.realpath(os.path.join(BMINDS_DATA_FOLDER, "BMCR_STPT_template"))
            BMINDS_ATLAS_FOLDER = os.path.realpath(os.path.join(BMINDS_TEMPLATES_FOLDER, "Atlases"))
            BMINDS_STPT_TEMPLATE_FOLDER = os.path.realpath(os.path.join(BMINDS_TEMPLATES_FOLDER, "STPT_population_average"))
            BMINDS_TRANSFORMS_FOLDER = os.path.realpath(os.path.join(BMINDS_TEMPLATES_FOLDER, "ANTS_transforms"))
            BMINDS_INJECTIONS_FOLDER = os.path.realpath(os.path.join(BMINDS_CORE_FOLDER, "processed_tracer_data"))
            BMINDS_UNZIPPED_DWI_FOLDER = os.path.realpath(os.path.join(BMINDS_OUTPUTS_DMRI_FOLDER, "dMRI_unzipped"))

    # Create main MRTRIX folder
    MAIN_MRTRIX_FOLDER_DMRI = os.path.join(BMINDS_OUTPUTS_DMRI_FOLDER, "MRTRIX")
    MAIN_MRTRIX_FOLDER_INJECTIONS = os.path.join(BMINDS_OUTPUTS_INJECTIONS_FOLDER, "MRTRIX")

    # Return folder names
    return (BMINDS_DATA_FOLDER, BMINDS_OUTPUTS_DMRI_FOLDER, BMINDS_OUTPUTS_INJECTIONS_FOLDER, BMINDS_CORE_FOLDER,
            BMINDS_DWI_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_TEMPLATES_FOLDER, BMINDS_ATLAS_FOLDER, BMINDS_STPT_TEMPLATE_FOLDER, 
            BMINDS_TRANSFORMS_FOLDER, BMINDS_INJECTIONS_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER, MAIN_MRTRIX_FOLDER_DMRI, 
            MAIN_MRTRIX_FOLDER_INJECTIONS)
       

# Check that output folders with subfolders are in suitable shape
def check_output_folders(folder, name, wipe=True, verbose=False):
    if not os.path.exists(folder):
        if verbose:
            print("--- {} folder not found. Created folder: {}".format(name, folder))
        os.makedirs(folder)
    # If it has content, delete it
    else:
        if wipe:
            if verbose:
                print("--- {} folder found. Continuing...".format(name))
            if len(os.listdir(folder)) != 0:
                if verbose:
                    print("{} folder has content. Deleting content...".format(name))
                # Since this can have both folders and files, we need to check if it's a file or folder to remove
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                if verbose:
                    print("Content deleted. Continuing...")
        else:
            if verbose:
                print("--- {} folder found. Continuing without wipe...".format(name))

# Retrieve (GLOB) files
def glob_files(PATH_NAME, file_format):
    INPUT_FILES = []
    for file in glob.glob(os.path.join(PATH_NAME, os.path.join("**", "*.{}".format(file_format))), recursive=True):
        INPUT_FILES.append(file)
    return INPUT_FILES

# Check that the retrieved (GLOBBED) files are not empty
def check_globbed_files(files, name, verbose=False):
    if len(files) == 0:
        print("No {} files found. Please add {} files".format(name, name))
        sys.exit('Exiting program')
    else:
        if verbose:
            print("{} files found. Continuing...".format(name))

# Check that input folders exist
def check_input_folders(folder, name, verbose=False):
    if not os.path.exists(folder):
        print("--- {} folder not found. Please add folder: {}".format(name, folder))
        sys.exit('Exiting program')
    else:
        if verbose:
            print("--- {} folder found. Continuing...".format(name))
        
