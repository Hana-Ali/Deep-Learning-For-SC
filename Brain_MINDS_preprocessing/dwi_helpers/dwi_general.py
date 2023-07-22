import os
import gzip
import tarfile
import magic
import numpy as np


from py_helpers import *

# First unzipping of all DWI files
def unzip_dwi_stage_1(ANIMAL_DWI_FILES, UNZIPPED_DWI_PATH):
    # For every file
    for file in ANIMAL_DWI_FILES:    
        with gzip.open(file, 'rb') as f_in:
            # Get the name of the region and file
            region_name = file.split(os.sep)[-2]
            file_name = file.split(os.sep)[-1]

            # Print what we're doing
            print("Unzipping Stage 1 for {}".format(region_name))

            # Create directory
            directory_to_extract_to = os.path.join(UNZIPPED_DWI_PATH, region_name)
            check_output_folders(directory_to_extract_to, region_name, wipe=False)
            
            # Add file name to directory
            file_in_directory = os.path.join(directory_to_extract_to, file_name)
            # If the file already exists, no need to unzip
            if os.path.exists(file_in_directory):
                print("File {} already exists. Continuing...".format(file_in_directory))
                continue
            # Unzip the file
            else:
                print("File {} does not exist. Unzipping...".format(file_in_directory))
                with open(file_in_directory, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

# Second unzipping of all DWI files
def unzip_dwi_stage_2(UNZIPPED_DWI_FILES, UNZIPPED_DWI_PATH): 
    for file in UNZIPPED_DWI_FILES:
        # Get the name of the region and file
        region_name = file.split(os.sep)[-2]
        file_name = file.split(os.sep)[-1].split(".")[0] + "_unzipped"

        # Print what we're doing
        print("Unzipping Stage 2 for {}".format(region_name))

        # Create directory
        directory_to_extract_to = os.path.join(UNZIPPED_DWI_PATH, region_name)
        check_output_folders(directory_to_extract_to, region_name, wipe=False)

        # Add file name to directory
        file_in_directory = os.path.join(directory_to_extract_to, file_name)
        print("File in directory: {}".format(file_in_directory))
        print("File we're unzipping: {}".format(file))

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

        print("Unzipping {} files".format(len(TO_UNZIP_DWI_FILES)))
        print("Unzipping {} files".format(TO_UNZIP_DWI_FILES))

        # Unzip the files
        unzip_dwi_stage_1(TO_UNZIP_DWI_FILES, BMINDS_UNZIPPED_DWI_FOLDER)
        UNZIPPED_STAGE_1 = glob_files(BMINDS_UNZIPPED_DWI_FOLDER, "nii.gz")
        check_globbed_files(UNZIPPED_STAGE_1, "UNZIPPED_STAGE_1")
        unzip_dwi_stage_2(UNZIPPED_STAGE_1, BMINDS_UNZIPPED_DWI_FOLDER) 

# Create a list that associates each subject with its T1 and DWI files
def create_data_list(BMINDS_DWI_FILES, BMINDS_BVAL_FILES, BMINDS_BVEC_FILES, BMINDS_STREAMLINE_FILES, 
                     BMINDS_INJECTION_FILES, BMINDS_ATLAS_FILE, BMINDS_ATLAS_LABEL_FILE, BMINDS_STPT_FILE, 
                     BMCR=True):
    DATA_LISTS = []
    RESIZED_DATA_LISTS = []
    EXTRACTED_DATA_LISTS = []

    # Get the initial lists
    (DWI_LIST, STREAMLINE_LIST, INJECTION_LIST) = create_initial_lists(BMINDS_DWI_FILES, BMINDS_BVAL_FILES,
                                                                        BMINDS_BVEC_FILES, BMINDS_STREAMLINE_FILES,
                                                                        BMINDS_INJECTION_FILES, BMCR=BMCR)
    
    # Join all DWIs with the same region name but different bvals and bvecs using mrtrix
    (CONCATENATED_DWI_LIST, RESIZED_CONCAT_DWI_LIST,
     EXTRACTED_DATA_LISTS) = join_dwi_diff_bvals_bvecs(DWI_LIST, BMCR=BMCR) 

    # Join the two lists based on common subject name
    DATA_LISTS = concatenate_common_subject_name(CONCATENATED_DWI_LIST, STREAMLINE_LIST, INJECTION_LIST,
                                                    BMINDS_ATLAS_FILE, BMINDS_ATLAS_LABEL_FILE, BMINDS_STPT_FILE,
                                                    BMCR=BMCR)
    if BMCR:
        # Do the same for the resized data
        RESIZED_DATA_LISTS = concatenate_common_subject_name(RESIZED_CONCAT_DWI_LIST, STREAMLINE_LIST, INJECTION_LIST,
                                                            BMINDS_ATLAS_FILE, BMINDS_ATLAS_LABEL_FILE, BMINDS_STPT_FILE,
                                                            BMCR=BMCR)
        # Do the same for the extracted data
        EXTRACTED_DATA_LISTS = concatenate_common_subject_name(EXTRACTED_DATA_LISTS, STREAMLINE_LIST, INJECTION_LIST,
                                                                BMINDS_ATLAS_FILE, BMINDS_ATLAS_LABEL_FILE, BMINDS_STPT_FILE,
                                                                BMCR=BMCR)
                
    return (DATA_LISTS, RESIZED_DATA_LISTS, EXTRACTED_DATA_LISTS)     

# Function to join based on common subject name
def concatenate_common_subject_name(DWI_LIST, STREAMLINE_LIST, INJECTION_LIST, BMINDS_ATLAS_FILE, 
                                    BMINDS_ATLAS_LABEL_FILE, BMINDS_STPT_FILE, BMCR=True):

    # Lists to hold the data
    DATA_LISTS = []

    # If resizing, join the two lists based on common subject name
    if BMCR:
        # For every dwi list
        for dwi_list in DWI_LIST:
            # Get the region, or common element ID
            region_ID = dwi_list[0].split(os.sep)[-3]
            # Based on this name, get every streamline and injection that has the same region ID
            streamline_files = [[streamline_file[1], streamline_file[2]] for streamline_file in STREAMLINE_LIST if streamline_file[0] == region_ID]
            injection_files = [[injection_file[1], injection_file[2]] for injection_file in INJECTION_LIST if injection_file[0] == region_ID]
            # Add the atlas and stpt files
            atlas_stpt = [BMINDS_ATLAS_FILE[0], BMINDS_ATLAS_LABEL_FILE[0], BMINDS_STPT_FILE[0]]
            # Extract the dwi, bval and bvec files
            dwi_data = [dwi_list[0], dwi_list[1], dwi_list[2], dwi_list[3]]

            # Check that streamline and injection files are not empty
            if not streamline_files:
                print("No streamline files found for {}".format(region_ID))
                continue
            if not injection_files:
                print("No injection files found for {}".format(region_ID))
                continue
            # Append the subject name, dwi, bval, bvec, streamline and injection to the list
            DATA_LISTS.append([region_ID, dwi_data, streamline_files, injection_files, atlas_stpt])
    # If not resizing - working with BMA data
    else:
        # For every dwi list
        for dwi_list in DWI_LIST:
            # Get the region ID
            region_ID = dwi_list[0].split(os.sep)[-1].split(".")[0]
            # Add the atlas and stpt files
            atlas_stpt = [BMINDS_ATLAS_FILE[0], BMINDS_ATLAS_LABEL_FILE[0], BMINDS_STPT_FILE[0]]
            # Extract the dwi, bval and bvec files
            dwi_data = [dwi_list[0], dwi_list[1], dwi_list[2], dwi_list[3]]

            # Append the subject name, dwi, bval, bvec, streamline and injection to the list
            DATA_LISTS.append([region_ID, dwi_data, None, None, atlas_stpt])

    return DATA_LISTS