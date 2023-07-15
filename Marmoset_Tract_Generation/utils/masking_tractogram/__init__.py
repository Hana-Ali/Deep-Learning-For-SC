"""
This folder will have the function of editing the image based on the tractogram being used for reference, where for the loss function
we want to only count the voxels for the areas that are overlapping between the two.

This will be done by using the tractogram as a mask for the image, where the voxels that are not in the tractogram will be set to 0. 
This will be done by using the function sitk.MaskImageFilter() from SimpleITK. This will be done in the __call__ function of the class 
MaskImage, where the tractogram will be used as the mask for the image. 

The tractogram will be converted to a binary image, where the voxels that are 1 will be the voxels that are in the tractogram, 
and the voxels that are 0 will be the voxels that are not in the tractogram. This will be done by using the function 
sitk.BinaryThreshold() from SimpleITK. The tractogram will be converted to a binary image in the __call__ function of the 
class BinaryThreshold, where the tractogram will be converted to a binary image
"""

# Import the masking function
from .masking import *