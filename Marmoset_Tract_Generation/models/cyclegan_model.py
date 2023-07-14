"""
This file is dedicated to the cycleGAN model. It uses the cycleGAN model to generate synthetic DWI images
from a given DWI image and a given tractogram, and the other way around
"""

import torch
from .base_models.base_model import BaseModel
from .network_helpers.network_funcs import *

