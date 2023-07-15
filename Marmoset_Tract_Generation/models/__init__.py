"""
This folder is dedicated to models. It includes everything in separate subfolders
"""

# Import the models and functions from this folder
from .model_options import *
from .model_builders import *

import importlib

# Function to find the model using name, for the training options
def find_model_using_name(model_name):
    # Define what the model file name is
    model_filename = "models." + model_name + "_model"
    # Import the model file
    modellib = importlib.import_module(model_filename)

    # Instantiate the model, which is a subclass of BaseModel
    model = None
    target_model_name = model_name.replace('_', '') + 'model'