import os
import glob
import sys
import shutil
import gzip
import tarfile
import magic

# Get the main paths
def get_main_paths(hpc):
    # Depending on whether we're in HPC or not, paths change
    if hpc == True:
        ALL_DATA_FOLDER = "/rds/general/user/hsa22/home/dissertation"
        SUBJECTS_FOLDER = "" # Empty in the case of HPC
        TRACTOGRAPHY_OUTPUT_FOLDER = os.path.join(ALL_DATA_FOLDER, "output_data")
        NIPYPE_OUTPUT_FOLDER = os.path.join(ALL_DATA_FOLDER, "nipype_outputs")
        FMRI_MAIN_FOLDER = os.path.join(ALL_DATA_FOLDER, "camcan_parcellated_acompcor/schaefer232/fmri700/rest")
        ATLAS_FOLDER = os.path.join(ALL_DATA_FOLDER, "atlas")

        PEDRO_MAIN_FOLDER = "/rds/general/user/pam213/home/Data/CAMCAN/"
        DWI_MAIN_FOLDER = os.path.join(PEDRO_MAIN_FOLDER, "dwi")
        T1_MAIN_FOLDER = os.path.join(PEDRO_MAIN_FOLDER, "aamod_dartel_norm_write_00001")
        
        DSI_COMMAND = "singularity exec dsistudio_latest.sif dsi_studio"

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
            BMINDS_DWI_FOLDER = os.path.realpath(os.path.join(BMINDS_DATA_FOLDER, "DWI_files"))
            BMINDS_METADATA_FOLDER = os.path.realpath(os.path.join(BMINDS_DATA_FOLDER, "meta_data"))
            BMINDS_INJECTIONS_FOLDER = os.path.realpath(os.path.join(BMINDS_DATA_FOLDER, "processed_tracer_data"))
            BMINDS_ATLAS_STPT_FOLDER = os.path.realpath(os.path.join(BMINDS_DATA_FOLDER, "atlas_stpt_template"))
            BMINDS_ZIPPED_DWI_FOLDER = os.path.realpath(os.path.join(BMINDS_DWI_FOLDER, "zipped_data"))
            BMINDS_UNZIPPED_DWI_FOLDER = os.path.realpath(os.path.join(BMINDS_DWI_FOLDER, "unzipped_data"))


    # Return folder names
    return (BMINDS_DATA_FOLDER, BMINDS_METADATA_FOLDER, BMINDS_INJECTIONS_FOLDER, BMINDS_ATLAS_STPT_FOLDER,
            BMINDS_ZIPPED_DWI_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER)
       

# Check that output folders with subfolders are in suitable shape
def check_output_folders(folder, name, wipe=True):
    if not os.path.exists(folder):
        print("--- {} folder not found. Created folder: {}".format(name, folder))
        os.makedirs(folder)
    # If it has content, delete it
    else:
        if wipe:
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
                print("Content deleted. Continuing...")
        else:
            print("--- {} folder found. Continuing without wipe...".format(name))

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

# Check that input folders exist
def check_input_folders(folder, name):
    if not os.path.exists(folder):
        print("--- {} folder not found. Please add folder: {}".format(name, folder))
        sys.exit('Exiting program')
    else:
        print("--- {} folder found. Continuing...".format(name))

# First unzipping of all DWI files
def unzip_dwi_stage_1(ANIMAL_DWI_FILES, UNZIPPED_DWI_PATH):
    # For every file
    for file in ANIMAL_DWI_FILES:    
        with gzip.open(file, 'rb') as f_in:
            # Get the name of the region and file
            if os.name == "nt":
                region_name = file.split("\\")[-2]
                file_name = file.split("\\")[-1]
            else:
                region_name = file.split("/")[-2]
                file_name = file.split("/")[-1]

            # Print what we're doing
            print("Unzipping Stage 1 for {}".format(region_name))

            # Create directory
            directory_to_extract_to = os.path.join(UNZIPPED_DWI_PATH, region_name)
            check_output_folders(directory_to_extract_to, region_name)
            
            # Add file name to directory
            file_in_directory = os.path.join(directory_to_extract_to, file_name)

            # Unzip the file
            with open(file_in_directory, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

# Second unzipping of all DWI files
def unzip_dwi_stage_2(UNZIPPED_DWI_FILES, UNZIPPED_DWI_PATH): 
    for file in UNZIPPED_DWI_FILES:
        # Get the name of the region and file
        if os.name == "nt":
            region_name = file.split("\\")[-2]
            file_name = file.split("\\")[-1].split(".")[0] + "_unzipped"
        else:
            region_name = file.split("/")[-2]
            file_name = file.split("/")[-1].split(".")[0] + "_unzipped"

        # Print what we're doing
        print("Unzipping Stage 2 for {}".format(region_name))

        # Create directory
        directory_to_extract_to = os.path.join(UNZIPPED_DWI_PATH, region_name)
        check_output_folders(directory_to_extract_to, region_name, wipe=False)

        # Add file name to directory
        file_in_directory = os.path.join(directory_to_extract_to, file_name)

        # Print the file type
        print(magic.from_file(file))

        # Unzip the file
        my_tar = tarfile.open(file)
        my_tar.extractall(file_in_directory)
        my_tar.close()

        # Delete the original .tar file and move the .nii file to the unzipped folder
        os.remove(file)
        move_data_from_deepest_folder(file_in_directory)


# Function to move data from the depths of the unzipped folder to a bit outside
def move_data_from_deepest_folder(file_in_directory):
    # Get the actual nii file from the depths of the unzipped folder
    nii_file = glob_files(file_in_directory, "nii")
    check_globbed_files(nii_file, "nii_file")
    # Move the file to the unzipped folder
    shutil.move(nii_file[0], file_in_directory)
    # Delete folder in this folder - which is the super long one
    for filename in os.listdir(file_in_directory):
        file_path = os.path.join(file_in_directory, filename)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)

# Function to check whether or not we need to do unzipping and unzips missing files
def check_unzipping(BMINDS_DWI_FOLDER, BMINDS_UNZIPPED_DWI_FOLDER):
    # Glob all the DWI and unzipped files
    BMINDS_DWI_FILES = glob_files(BMINDS_DWI_FOLDER, "nii.gz")
    BMINDS_UNZIPPED_DWI_FILES = glob_files(BMINDS_UNZIPPED_DWI_FOLDER, "nii")
    # Check globbed inputs
    check_globbed_files(BMINDS_DWI_FILES, "BMINDS_DWI_FILES")

    # Get the folder names for both DWI and unzipped DWI to compare
    BMINDS_DWI_FOLDER_NAMES = [file.split("/")[-2] for file in BMINDS_DWI_FILES]
    UNZIPPED_DWI_FOLDER_NAMES = [file.split("/")[-3] for file in BMINDS_UNZIPPED_DWI_FILES]

    print("BMINDS_DWI_FOLDER_NAMES: {}".format(BMINDS_DWI_FOLDER_NAMES))
    print("UNZIPPED_DWI_FOLDER_NAMES: {}".format(UNZIPPED_DWI_FOLDER_NAMES))

    # Check whether all folder names in DWI are in unzipped DWI
    if not all(folder in UNZIPPED_DWI_FOLDER_NAMES for folder in BMINDS_DWI_FOLDER_NAMES):
        # Get the missing files we need to unzip
        TO_UNZIP_DWI_FILES = [file for file in BMINDS_DWI_FILES if file.split("/")[-2] not in UNZIPPED_DWI_FOLDER_NAMES]

        print("BMINDS_DWI_FILES: {}".format(BMINDS_DWI_FILES))
        print("BMINDS_UNZIPPED_DWI_FILES: {}".format(BMINDS_UNZIPPED_DWI_FILES))
        print("Unzipping {} files".format(len(TO_UNZIP_DWI_FILES)))
        print("Unzipping {} files".format(TO_UNZIP_DWI_FILES))

        # Unzip the files
        unzip_dwi_stage_1(TO_UNZIP_DWI_FILES, BMINDS_UNZIPPED_DWI_FOLDER)
        UNZIPPED_STAGE_1 = glob_files(BMINDS_UNZIPPED_DWI_FOLDER, "nii.gz")
        check_globbed_files(UNZIPPED_STAGE_1, "UNZIPPED_STAGE_1")
        unzip_dwi_stage_2(UNZIPPED_STAGE_1, BMINDS_UNZIPPED_DWI_FOLDER) 

# Function to extract the DWI filename
def extract_region_ID(file):
    # Extract the filename
    if os.name == "nt":
        region_name = file.split("\\")[-3]
    else:
        region_name = file.split("/")[-3]
    # Return the filename
    return region_name

# Function to extract the BVAL filename
def extract_correct_bval(dwi_file, BMINDS_BVAL_FILES):
    # Extract the correct number for the bval
    if os.name == "nt":
        bval_name = dwi_file.split("\\")[-1]
    else:
        bval_name = dwi_file.split("/")[-1].split(".")[0]
    
    # For all the bvals extracted
    for bval_file in BMINDS_BVAL_FILES:
        # If the bval file has the same name as the bval name (i.e. the number 1000, 3000, etc.)
        if bval_name in bval_file:
            # Return the bval filepath
            return bval_file
    
    # If we don't find the bval file, exit the program
    print("No bval file found for {}".format(dwi_file))
    sys.exit('Exiting program')

# Function to extract the BVEC filename
def extract_correct_bvec(dwi_file, BMINDS_BVEC_FILES):
    # Extract the correct number for the bval
    if os.name == "nt":
        bvec_name = dwi_file.split("\\")[-1]
    else:
        bvec_name = dwi_file.split("/")[-1].split(".")[0]
    
    # For all the bvecs extracted
    for bvec_file in BMINDS_BVEC_FILES:
        # If the bvec file has the same name as the bvec name (i.e. the number 1000, 3000, etc.)
        if bvec_name in bvec_file:
            # Return the bvec filepath
            return bvec_file
    
    # If we don't find the bvec file, exit the program
    print("No bvec file found for {}".format(dwi_file))
    sys.exit('Exiting program')

# Function to determine what type of streamline file it is (dwi, tract-tracing, etc)
def get_streamline_type(file):
    # Extract the filename
    if os.name == "nt":
        streamline_name = file.split("\\")[-1].split(".")[0]
    else:
        streamline_name = file.split("/")[-1].split(".")[0]
    
    # Return different things, depending on the name
    if 'sharp' in streamline_name:
        return 'tracer_sharp'
    elif 'tracer' in streamline_name:
        return 'tracer'
    elif 'track' in streamline_name:
        return 'dwi'
    # If none of the above, return unknown error
    else:
        print("Unknown streamline file type for {}. Name is {}".format(file, streamline_name))
        sys.exit('Exiting program')

# Function to determine what type of injection file it is (density, etc)
def get_injection_type(file):
    # Extract the filename
    if os.name == "nt":
        injection_name = file.split("\\")[-1].split(".")[0]
    else:
        injection_name = file.split("/")[-1].split(".")[0]
    
    # Return different things, depending on the name
    if 'cell_density' in injection_name:
        return 'cell_density'
    elif 'positive_voxels' in injection_name:
        return 'tracer_positive_voxels'
    elif 'signal_normalized' in injection_name:
        return 'tracer_signal_normalized'
    elif 'signal' in injection_name:
        return 'tracer_signal'
    elif 'density' in injection_name:
        return 'streamline_density'
    # If none of the above, return unknown error
    else:
        print("Unknown injection file type for {}. Name is {}".format(file, injection_name))
        sys.exit('Exiting program')

# Create a list that associates each subject with its T1 and DWI files
def create_data_list(BMINDS_UNZIPPED_DWI_FILES, BMINDS_BVAL_FILES, BMINDS_BVEC_FILES, BMINDS_STREAMLINE_FILES, BMINDS_INJECTION_FILES):
    DATA_LISTS = {}
    DWI_LIST = []
    STREAMLINE_LIST = {}
    INJECTION_LIST = {}

    # For each DWI file
    for dwi_file in BMINDS_UNZIPPED_DWI_FILES:
        # Get the region ID
        region_ID = extract_region_ID(dwi_file)

        # Get the bval and bvec files
        bval_path = extract_correct_bval(dwi_file, BMINDS_BVAL_FILES)
        bvec_path = extract_correct_bvec(dwi_file, BMINDS_BVEC_FILES)

        # Append to a DWI list
        DWI_LIST.append([region_ID, dwi_file, bval_path, bvec_path])

    # For each streamline file
    for streamline_file in BMINDS_STREAMLINE_FILES:
        # Get the region ID
        region_ID = extract_region_ID(streamline_file)
        # Get the type of streamline file it is
        streamline_type = get_streamline_type(streamline_file)
        # Append all the data to the dictionary
        STREAMLINE_LIST[region_ID] = STREAMLINE_LIST.get(region_ID, {}) | {streamline_type: streamline_file}

    # For each injection file
    for injection_file in BMINDS_INJECTION_FILES:
        # Get the region ID
        region_ID = extract_region_ID(injection_file)
        # Get the type of injection file it is
        injection_type = get_injection_type(injection_file)
        # Append all the data to the dictionary
        INJECTION_LIST[region_ID] = INJECTION_LIST.get(region_ID, {}) | {injection_type: injection_file}

    # Join the two lists based on common subject name
    for dwi in DWI_LIST:
        # Get the region, or common element ID
        region_ID = dwi[0]
        # Based on this name, get every streamline and injection that has the same region ID
        streamline_files = STREAMLINE_LIST[region_ID]
        injection_files = INJECTION_LIST[region_ID]
        # Check that streamline and injection files are not empty
        if not streamline_files:
            print("No streamline files found for {}".format(region_ID))
            continue
        if not injection_files:
            print("No injection files found for {}".format(region_ID))
            continue
        # Append the subject name, dwi, bval, bvec, streamline and injection to the list
        DATA_LISTS[region_ID] = {'dwi_file': dwi[1], 'bval_path': dwi[2], 'bvec_path': dwi[3], 
                                 'streamline_files': streamline_files, 'injection_files': injection_files}
            
    return DATA_LISTS


# # Function to get the streamline filename
# def extract_streamline_region(file):
#     # Extract the filename
#     if os.name == "nt":
#         region_name = file.split("\\")[-3]
#     else:
#         region_name = file.split("/")[-3]
#     # Return the filename
#     return region_name

# # Function to get the injection filename
# def extract_injection_region(file):
#     # Extract the filename
#     if os.name == "nt":
#         region_name = file.split("\\")[-3]
#     else:
#         region_name = file.split("/")[-3]
#     # Return the filename
#     return region_name
