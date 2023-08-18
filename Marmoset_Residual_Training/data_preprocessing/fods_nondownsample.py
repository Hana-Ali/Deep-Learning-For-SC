import sys
from py_helpers.general_helpers import *
import numpy as np

import subprocess


# Function to define the paths to the data
def define_paths(DWI_LIST):

    ##### MASK
    MIF_DWI = DWI_LIST[1].replace(".nii", ".mif")
    MIF_MASK = MIF_DWI.replace(".mif", "_mask.mif")
    NII_MASK = MIF_MASK.replace(".mif", ".nii")

    ##### CLEAN PATHS
    DENOISE_DWI = MIF_DWI.replace(".mif", "_denoised.mif")
    NOISE_MAP = MIF_DWI.replace(".mif", "_noise.mif")
    BIAS_CORRECTED_DWI = MIF_DWI.replace(".mif", "_bias.mif")

    ##### RESPONSE
    RESPONSE_VOXELS_TXT = MIF_DWI.replace(".mif", "_responsevoxels.txt")
    WM_TXT = MIF_DWI.replace(".mif", "_wm.txt")
    GM_TXT = MIF_DWI.replace(".mif", "_gm.txt")
    CSF_TXT = MIF_DWI.replace(".mif", "_csf.txt")

    ##### FODS
    WMFOD_MIF = MIF_DWI.replace(".mif", "_wmfod.mif")
    GMFOD_MIF = MIF_DWI.replace(".mif", "_gmfod.mif")
    CSFFOD_MIF = MIF_DWI.replace(".mif", "_csffod.mif")
    VFFOD_MIF = MIF_DWI.replace(".mif", "_vffod.mif")

    ##### NORMALIZE FODS
    WMFOD_NORM = MIF_DWI.replace(".mif", "_wmfod_norm.mif")
    GMFOD_NORM = MIF_DWI.replace(".mif", "_gmfod_norm.mif")
    CSFFOD_NORM = MIF_DWI.replace(".mif", "_csffod_norm.mif")

    # Return the paths
    return (MIF_DWI, MIF_MASK, NII_MASK, DENOISE_DWI, NOISE_MAP, 
            BIAS_CORRECTED_DWI, RESPONSE_VOXELS_TXT, WM_TXT, GM_TXT, 
            CSF_TXT, WMFOD_MIF, GMFOD_MIF, CSFFOD_MIF, VFFOD_MIF, WMFOD_NORM, 
            GMFOD_NORM, CSFFOD_NORM)

# Function that does the operations we need
def get_fods(DWI_LIST):

    # Get the paths
    (MIF_DWI, MIF_MASK, NII_MASK, DENOISE_DWI, NOISE_MAP,
        BIAS_CORRECTED_DWI, RESPONSE_VOXELS_TXT, WM_TXT, GM_TXT,
        CSF_TXT, WMFOD_MIF, GMFOD_MIF, CSFFOD_MIF, VFFOD_MIF, WMFOD_NORM,
        GMFOD_NORM, CSFFOD_NORM) = define_paths(DWI_LIST)
    
    MIF_CNVRT_CMD = "mrconvert {input} -fslgrad {bvec} {bval} {output}".format(input=DWI_LIST[1], bvec=DWI_LIST[2], bval=DWI_LIST[3], output=MIF_DWI)
    MASK_CMD = "dwi2mask {input} {output}".format(input=MIF_DWI, output=MIF_MASK)
    NII_MASK_CMD = "mrconvert {input} {output}".format(input=MIF_MASK, output=NII_MASK)

    DENOISE_CMD = "dwidenoise {input} {output} -noise {noise_map}".format(input=MIF_DWI, output=DENOISE_DWI, noise_map=NOISE_MAP)
    BIAS_CORRECT_CMD = "dwibiascorrect ants {input} {output} -mask {mask}".format(input=DENOISE_DWI, output=BIAS_CORRECTED_DWI, mask=MIF_MASK)

    RESPONSE_CMD = "dwi2response dhollander {input} -mask {mask} {wm} {gm} {csf} -voxels \
        {voxels}".format(input=BIAS_CORRECTED_DWI, mask=MIF_MASK, wm=WM_TXT, gm=GM_TXT, csf=CSF_TXT, voxels=RESPONSE_VOXELS_TXT)
    MULTISHELL_CSD_CMD = "dwi2fod msmt_csd {input} {wm} {wmfod} {gm} {gmfod} {csf} {csffod} \
        -mask {mask} -force".format(input=BIAS_CORRECTED_DWI, wm=WM_TXT, wmfod=WMFOD_MIF, gm=GM_TXT, gmfod=GMFOD_MIF, csf=CSF_TXT, csffod=CSFFOD_MIF, mask=MIF_MASK)
    COMBINE_FODS_CMD = "mrconvert -coord 3 0 {wmfod}.mif - | mrcat {csffod}.mif {gmfod}.mif - {output}.mif".format(
        wmfod=WMFOD_MIF, csffod=CSFFOD_MIF, gmfod=GMFOD_MIF, output=VFFOD_MIF)
    NORMALIZE_FODS_CMD = "mtnormalise {wmfod} {wmfod_norm} {gmfod} {gmfod_norm} {csffod} {csffod_norm} \
        -mask {mask} -force".format(wmfod=WMFOD_MIF, wmfod_norm=WMFOD_NORM, gmfod=GMFOD_MIF, gmfod_norm=GMFOD_NORM, csffod=CSFFOD_MIF, csffod_norm=CSFFOD_NORM, mask=MIF_MASK)

    # Run the commands
    if not os.path.exists(MIF_DWI):
        print("Running: {}".format(MIF_CNVRT_CMD))
        subprocess.call(MIF_CNVRT_CMD, shell=True)
    
    if not os.path.exists(MIF_MASK):
        print("Running: {}".format(MASK_CMD))
        subprocess.call(MASK_CMD, shell=True)

    if not os.path.exists(NII_MASK):
        print("Running: {}".format(NII_MASK_CMD))
        subprocess.call(NII_MASK_CMD, shell=True)

    if not os.path.exists(DENOISE_DWI):
        print("Running: {}".format(DENOISE_CMD))
        subprocess.call(DENOISE_CMD, shell=True)

    if not os.path.exists(BIAS_CORRECTED_DWI):
        print("Running: {}".format(BIAS_CORRECT_CMD))
        subprocess.call(BIAS_CORRECT_CMD, shell=True)

    if not os.path.exists(RESPONSE_VOXELS_TXT):
        print("Running: {}".format(RESPONSE_CMD))
        subprocess.call(RESPONSE_CMD, shell=True)

    if not os.path.exists(VFFOD_MIF):
        print("Running: {}".format(MULTISHELL_CSD_CMD))
        subprocess.call(MULTISHELL_CSD_CMD, shell=True)

    if not os.path.exists(WMFOD_NORM):
        print("Running: {}".format(NORMALIZE_FODS_CMD))
        subprocess.call(NORMALIZE_FODS_CMD, shell=True)
    

    

# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    if hpc:
        data_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/processed_dMRI/extracted_fullsize_dwi"
    else:
        data_path = "/media/hsa22/Expansion/Brain-MINDS/processed_dMRI/extracted_fullsize_dwi"

    # Grab all the nii, bvecs and bval files
    nii_files = glob_files(data_path, "nii")
    bvec_files = glob_files(data_path, "bvecs")
    bval_files = glob_files(data_path, "bvals")

    # Create list of [region_ID, nii, bvec, bval]
    nii_bvec_bval = []
    for nii in nii_files:
        region_ID = nii.split(os.sep)[-2]
        bvec = [bvec for bvec in bvec_files if region_ID in bvec][0]
        bval = [bval for bval in bval_files if region_ID in bval][0]
        nii_bvec_bval.append([region_ID, nii, bvec, bval])

    print("Found {} nii files".format(len(nii_bvec_bval)))

    # Run the commands
    region_idx = int(sys.argv[2])
    get_fods(nii_bvec_bval[region_idx])


if __name__ == "__main__":
    main()
