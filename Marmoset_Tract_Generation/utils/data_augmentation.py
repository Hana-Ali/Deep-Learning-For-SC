"""
This file is dedicated to data augmentation.
Specifically, we want to be able to augment the DWI data in a way that is consistent with the
augmentation of the corresponding tractogram.

The augmentation methods are:
    - Random rotation
    - Random translation
    - Random scaling
    - Random flipping
    - Random noise
    - Random cropping

The augmentation methods are implemented as functions that take in a DWI image and a tractogram and
return the augmented DWI image and tractogram.
"""

import numpy as np
import SimpleITK as sitk

