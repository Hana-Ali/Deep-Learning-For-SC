import sys
from py_helpers.general_helpers import *
import numpy as np

import nibabel as nib

from nibabel.streamlines import Field
from nibabel.orientations import aff2axcodes

import subprocess

# Define MRtrix global tracking paths
def get_mrtrix_global_tracking_paths(report_plots_path, optimized=False):

    # Create the global tractography folder
    GLOBAL_TRACKING_FOLDER_NAME = os.path.join(report_plots_path, "global_tractography")

    # If optimized, then add the optimized folder
    if optimized:
        GLOBAL_TRACKING_FOLDER_NAME = os.path.join(GLOBAL_TRACKING_FOLDER_NAME, "optimized")
    else:
        GLOBAL_TRACKING_FOLDER_NAME = os.path.join(GLOBAL_TRACKING_FOLDER_NAME, "unoptimized")

    # Check the output folders exist
    check_output_folders(GLOBAL_TRACKING_FOLDER_NAME, "global_tractography", wipe=False)
    
    # FOD and FISO for global tractography paths
    GLOBAL_FOD_PATH = os.path.join(GLOBAL_TRACKING_FOLDER_NAME, "global_fod")
    GLOBAL_FISO_PATH = os.path.join(GLOBAL_TRACKING_FOLDER_NAME, "global_fiso")
    # Global tractography path
    GLOBAL_TRACT_PATH = os.path.join(GLOBAL_TRACKING_FOLDER_NAME, "global_tract")

    # Return the paths
    return (GLOBAL_FOD_PATH, GLOBAL_FISO_PATH, GLOBAL_TRACT_PATH)

# Define MRtrix probabilistic tracking paths
def get_mrtrix_probtrack_paths(report_plots_path, optimized=False):

   # Create the global tractography folder
    PROBABILISTIC_TRACTOGRAPHY_PATH = os.path.join(report_plots_path, "prob_tractography")

    # If optimized, then add the optimized folder
    if optimized:
        PROBABILISTIC_TRACTOGRAPHY_PATH = os.path.join(PROBABILISTIC_TRACTOGRAPHY_PATH, "optimized")
    else:
        PROBABILISTIC_TRACTOGRAPHY_PATH = os.path.join(PROBABILISTIC_TRACTOGRAPHY_PATH, "unoptimized")

    # Check the output folders exist
    check_output_folders(PROBABILISTIC_TRACTOGRAPHY_PATH, "probabilistic_tractography", wipe=False)

    # Mask for streamline seeding paths and probabilistic tractography path
    GM_WM_SEED_PATH = os.path.join(PROBABILISTIC_TRACTOGRAPHY_PATH, "gmwmseed")
    TRACT_TCK_PATH = os.path.join(PROBABILISTIC_TRACTOGRAPHY_PATH, "prob_tract")

    # Return the paths
    return (GM_WM_SEED_PATH, TRACT_TCK_PATH)

# Function to define the response paths
def get_mrtrix_response_paths(data_path):

    # Define the general and response and fod_norm folders
    GENERAL_FOLDER_NAME = os.path.join(data_path, "general")
    RESPONSE_FOLDER_NAME = os.path.join(data_path, "response")
    FOD_NORM_FOLDER_NAME = os.path.join(data_path, "fod_norm")

    # Define the clean input and mask mif path
    INPUT_MIF_PATH = os.path.join(GENERAL_FOLDER_NAME, "A6DR-R01_0083-TT56_clean_mif")
    MASK_NII_PATH = os.path.join(GENERAL_FOLDER_NAME, "A6DR-R01_0083-TT56_clean_mask_nii")
    MASK_MIF_PATH = os.path.join(GENERAL_FOLDER_NAME, "A6DR-R01_0083-TT56_clean_mask")

    # Define the response paths
    RESPONSE_WM_PATH = os.path.join(RESPONSE_FOLDER_NAME, "A6DR-R01_0083-TT56_wm")
    RESPONSE_CSF_PATH = os.path.join(RESPONSE_FOLDER_NAME, "A6DR-R01_0083-TT56_csf")
    RESPONSE_GM_PATH = os.path.join(RESPONSE_FOLDER_NAME, "A6DR-R01_0083-TT56_gm")

    # Define the fod_norm paths
    WM_FOD_NORM_PATH = os.path.join(FOD_NORM_FOLDER_NAME, "A6DR-R01_0083-TT56_wmfod_norm")

    # Define the bval and bvec paths from the general folder
    BVAL_PATH = os.path.join(GENERAL_FOLDER_NAME, "A6DR-R01_0083-TT56_clean_bval.bval")
    BVEC_PATH = os.path.join(GENERAL_FOLDER_NAME, "A6DR-R01_0083-TT56_clean_bvec.bvec")

    # Return the paths
    return (INPUT_MIF_PATH, MASK_NII_PATH, MASK_MIF_PATH, RESPONSE_WM_PATH, 
            RESPONSE_CSF_PATH, RESPONSE_GM_PATH, WM_FOD_NORM_PATH, 
            BVAL_PATH, BVEC_PATH)

# Define the connectome paths
def get_connectome_paths(report_plots_path, optimized=False, track_name="prob", atlas_name="MBCA"):

    # Create the connectome folder
    CONNECTOME_FOLDER_NAME = os.path.join(report_plots_path, "connectome")

    # If optimized, then add the optimized folder
    if optimized:
        CONNECTOME_FOLDER_NAME = os.path.join(CONNECTOME_FOLDER_NAME, "optimized")
    else:
        CONNECTOME_FOLDER_NAME = os.path.join(CONNECTOME_FOLDER_NAME, "unoptimized")

    # Check the output folders exist
    check_output_folders(CONNECTOME_FOLDER_NAME, "connectome", wipe=False)

    # Define the connectome path
    CONNECTOME_PATH = os.path.join(CONNECTOME_FOLDER_NAME, "{}_{}_connectome".format(track_name, atlas_name))

    # Return the paths
    return CONNECTOME_PATH

# Function to perform probabilistic tractography
def probabilistic_tractography(report_plots_path, TRACT_TCK_PATH, MBCA_atlas_path, MBM_atlas_path, optimized=False):

    CONNECTOME_PATH = get_connectome_paths(report_plots_path, optimized=optimized, track_name="prob", atlas_name="MBCA")
    if not os.path.exists(CONNECTOME_PATH + ".csv"):
        CONNECTIVITY_CMD = "tck2connectome {tract}.tck {atlas} {connectome}.csv -zero_diagonal -symmetric \
            -assignment_all_voxels -force".format(tract=TRACT_TCK_PATH, 
                                                    atlas=MBCA_atlas_path, 
                                                    connectome=CONNECTOME_PATH)
        print("Creating connectome, optimized={}, track_name=prob, atlas_name=MBCA".format(optimized))
        subprocess.run(CONNECTIVITY_CMD, shell=True, check=True)
    else:
        print("Connectome file for optimized={}, track_name=prob, atlas_name=MBCA".format(optimized))
    # Probabilistic tractography - MBM
    CONNECTOME_PATH = get_connectome_paths(report_plots_path, optimized=optimized, track_name="prob", atlas_name="MBM")
    if not os.path.exists(CONNECTOME_PATH + ".csv"):
        CONNECTIVITY_CMD = "tck2connectome {tract}.tck {atlas} {connectome}.csv -zero_diagonal -symmetric \
            -assignment_all_voxels -force".format(tract=TRACT_TCK_PATH, 
                                                    atlas=MBM_atlas_path, 
                                                    connectome=CONNECTOME_PATH)
        print("Creating connectome, optimized={}, track_name=prob, atlas_name=MBM".format(optimized))
        subprocess.run(CONNECTIVITY_CMD, shell=True, check=True)

# Function to perform global tractography
def global_tractography(report_plots_path, GLOBAL_TRACT_PATH, MBCA_atlas_path, MBM_atlas_path, optimized=False):

        CONNECTOME_PATH = get_connectome_paths(report_plots_path, optimized=optimized, track_name="global", atlas_name="MBCA")
        if not os.path.exists(CONNECTOME_PATH + ".csv"):
            CONNECTIVITY_CMD = "tck2connectome {tract}.tck {atlas} {connectome}.csv -zero_diagonal -symmetric \
                -assignment_all_voxels -force".format(tract=GLOBAL_TRACT_PATH, 
                                                     atlas=MBCA_atlas_path, 
                                                     connectome=CONNECTOME_PATH)
            print("Creating connectome, optimized={}, track_name=global, atlas_name=MBCA".format(optimized))
            subprocess.run(CONNECTIVITY_CMD, shell=True, check=True)
        else:
            print("Connectome file for optimized={}, track_name=global, atlas_name=MBCA".format(optimized))
        # Global tractography - MBM
        CONNECTOME_PATH = get_connectome_paths(report_plots_path, optimized=optimized, track_name="global", atlas_name="MBM")
        if not os.path.exists(CONNECTOME_PATH + ".csv"):
            CONNECTIVITY_CMD = "tck2connectome {tract}.tck {atlas} {connectome}.csv -zero_diagonal -symmetric \
                -assignment_all_voxels -force".format(tract=GLOBAL_TRACT_PATH, 
                                                     atlas=MBM_atlas_path, 
                                                     connectome=CONNECTOME_PATH)
            print("Creating connectome, optimized={}, track_name=global, atlas_name=MBM".format(optimized))
            subprocess.run(CONNECTIVITY_CMD, shell=True, check=True)


# Function to do the streamline node extraction
def convert_to_trk(tck_file, template):

    # Open dwi
    nii = nib.load(template)
    
    # Get the filename of the streamline
    streamline_filename = tck_file.split(os.sep)[-1].replace(".tck", ".trk")

    # Define the output folder to be the same folder as the tck file
    output_folder = (os.sep).join(tck_file.split(os.sep)[:-1])

    # Define the new filepath
    new_filepath = os.path.join(output_folder, streamline_filename)

    # Load the tck
    tck = nib.streamlines.load(tck_file)

    # Make the header
    header = {}
    header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
    header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = nii.shape[:3]
    header[Field.VOXEL_ORDER] = "".join(aff2axcodes(nii.affine))

    # Save the new streamlines
    nib.streamlines.save(tck.tractogram, new_filepath, header=header)
    print("Saved new streamlines to {}".format(new_filepath))


# Main function
def main():

    hpc = int(sys.argv[1])
    if hpc:
        # Define the main data path
        data_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/processed_dMRI/MRTRIX/A6DR-R01_0083-TT56"
        template = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/model_data_w_resize/dMRI_b0/A6DR-R01_0083-TT56/DWI_concatenated_b0_resized.nii.gz"
        report_plots_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/processed_dMRI/MRTRIX/A6DR-R01_0083-TT56/report_plots"
        MBCA_atlas_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/BMCR_STPT_template/Atlases/MBCA_mapped/atlas_segmentation.nii.gz"
        MBM_atlas_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/BMCR_STPT_template/Atlases/MBM_mapped/MBM_cortex_vM_80um_TC_std.nii.gz"
    else:
        # Define the main data path
        data_path = "/mnt/d/Brain-MINDS/processed_dMRI/MRTRIX/A6DR-R01_0083-TT56"
        template = "/mnt/d/Brain-MINDS/model_data_w_resize/dMRI_b0/A6DR-R01_0083-TT56/DWI_concatenated_b0.nii.gz"
        report_plots_path = "/mnt/d/Brain-MINDS/processed_dMRI/MRTRIX/A6DR-R01_0083-TT56/report_plots"
        MBCA_atlas_path = "/mnt/d/Brain-MINDS/BMCR_STPT_template/Atlases/MBCA_mapped/atlas_segmentation.nii.gz"
        MBM_atlas_path = "/mnt/d/Brain-MINDS/BMCR_STPT_template/Atlases/MBM_mapped/MBM_cortex_vM_80um_TC_std.nii.gz"

    # Function to get paths of things given the data path
    (INPUT_MIF_PATH, MASK_NII_PATH, MASK_MIF_PATH, RESPONSE_WM_PATH,
     RESPONSE_CSF_PATH, RESPONSE_GM_PATH, WM_FOD_NORM_PATH,
     BVAL_PATH, BVEC_PATH) = get_mrtrix_response_paths(data_path)
    
    # Define the optimized and unoptimized booleans
    optimized_bools = [True, False]
    
    # For both optimized and unoptimized
    for optimized in optimized_bools:

        # 1. Probabilistic tractography
        # Get the probabilistic tractography paths
        (GM_WM_SEED_PATH, TRACT_TCK_PATH) = get_mrtrix_probtrack_paths(report_plots_path, optimized=optimized)

        # Probabilistic tractography command
        if optimized:
            PROB_TRACT_CMD = "tckgen {wmfod_norm}.mif {output}.tck -algorithm iFOD2 -seed_image {mask}.nii.gz -mask {mask}.nii.gz \
                -angle {opt_angle} -minlength {opt_minlength} -cutoff {opt_cutoff} \
                -fslgrad {bvec} {bval} -select 300000 -force".format(wmfod_norm=WM_FOD_NORM_PATH, output=TRACT_TCK_PATH, 
                                                                    mask=MASK_NII_PATH,
                                                                    opt_angle=32.2, opt_cutoff=0.05, opt_minlength=4.8,
                                                                    bvec=BVEC_PATH, bval=BVAL_PATH)
        else:
            # Probabilistic tractography command
            PROB_TRACT_CMD = "tckgen {wmfod_norm}.mif {output}.tck -algorithm iFOD2 -seed_image {mask}.nii.gz -mask {mask}.nii.gz \
                -fslgrad {bvec} {bval} -select 300000 -force".format(wmfod_norm=WM_FOD_NORM_PATH, output=TRACT_TCK_PATH, 
                                                                    mask=MASK_NII_PATH,
                                                                    opt_angle=32.2, opt_cutoff=0.05, opt_minlength=4.8,
                                                                    bvec=BVEC_PATH, bval=BVAL_PATH)
            
        # Run the probabilistic tractography command IF the file doesn't exist
        if not os.path.exists(TRACT_TCK_PATH + ".tck"):
            print("Running probabilistic tractography, optimized={}".format(optimized))
            subprocess.run(PROB_TRACT_CMD, shell=True, check=True)
        else:
            print("Probabilistic tractography file already exists, skipping")
            
        # 2. Global tractography
        # Get the global tractography paths
        (GLOBAL_FOD_PATH, GLOBAL_FISO_PATH, GLOBAL_TRACT_PATH) = get_mrtrix_global_tracking_paths(report_plots_path, optimized=optimized)

        # Global tractography command
        if optimized:
                GLOBAL_TRACT_CMD = "tckglobal {dwi}.mif {wm_response}.txt -riso {csf_response}.txt -riso \
                        {gm_response}.txt -mask {mask}.mif -niter 1e9 -fod {gt_fod}.mif \
                        -length {length} -weight {weight} -cpot {cpot} \
                        -fiso {gt_fiso}.mif {output}.tck".format(dwi=INPUT_MIF_PATH, wm_response=RESPONSE_WM_PATH, 
                                                                     csf_response=RESPONSE_CSF_PATH, gm_response=RESPONSE_GM_PATH,
                                                                     length=0.45, weight=0.054, cpot=0.106,
                                                                     mask=MASK_MIF_PATH, gt_fod=GLOBAL_FOD_PATH, 
                                                                     gt_fiso=GLOBAL_FISO_PATH, output=GLOBAL_TRACT_PATH)
        else:
                GLOBAL_TRACT_CMD = "tckglobal {dwi}.mif {wm_response}.txt -riso {csf_response}.txt -riso \
                        {gm_response}.txt -mask {mask}.mif -niter 1e9 \
                        -fod {gt_fod}.mif -fiso {gt_fiso}.mif {output}.tck".format(dwi=INPUT_MIF_PATH, wm_response=RESPONSE_WM_PATH, 
                                                                                       csf_response=RESPONSE_CSF_PATH, gm_response=RESPONSE_GM_PATH,
                                                                                       mask=MASK_MIF_PATH, gt_fod=GLOBAL_FOD_PATH, 
                                                                                       gt_fiso=GLOBAL_FISO_PATH, output=GLOBAL_TRACT_PATH)
        
        # Run the global tractography command IF the file doesn't exist
        if not os.path.exists(GLOBAL_TRACT_PATH + ".tck"):
            print("Running global tractography, optimized={}".format(optimized))
            subprocess.run(GLOBAL_TRACT_CMD, shell=True, check=True)
        else:
            print("Global tractography file already exists, skipping")

        # 3. Convert the tck files to trk files
        # Probabilistic tractography
        convert_to_trk(TRACT_TCK_PATH + ".tck", template)
        # Global tractography
        convert_to_trk(GLOBAL_TRACT_PATH + ".tck", template)

        # 4. Create the connectomes
        # Probabilistic tractography
        probabilistic_tractography(report_plots_path, TRACT_TCK_PATH, MBCA_atlas_path, MBM_atlas_path, optimized=optimized)

        # Global tractography - MBCA
        global_tractography(report_plots_path, GLOBAL_TRACT_PATH, MBCA_atlas_path, MBM_atlas_path, optimized=optimized)
        
        print("-"*50)
        print("Finished optimized={}".format(optimized))
        print("-"*50)


if __name__ == "__main__":
    main()