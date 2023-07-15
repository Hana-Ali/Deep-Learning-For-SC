"""
Experimenting with the masking of tractograms
"""

import SimpleITK as sitk
import os

from .masking_tractogram.masking import *

if __name__ == "__main__":
     
    # Define the paths to the image and tractogram
    main_data_path = "/mnt/c/tractography/combined_normal_test"
    combined_tract_path = os.path.join(main_data_path, "combined_tracts.nii.gz")
    individual_tract_path = os.path.join(main_data_path, "tracer_streamlines.nii.gz")

    # Define the output paths
    combined_tract_masked_path = os.path.join(main_data_path, "combined_masked.nii.gz")

    # Load the image and tractogram
    combined_tract = sitk.ReadImage(combined_tract_path)
    individual_tract = sitk.ReadImage(individual_tract_path)

    # Mask the image with the tractogram
    combined_tract_masked = MaskImage(individual_tract)(combined_tract)

    # Save the masked image
    sitk.WriteImage(combined_tract_masked, combined_tract_masked_path)

