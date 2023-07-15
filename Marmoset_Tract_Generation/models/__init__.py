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
    model_filename = "models.model_options." + model_name + "_model"
    # Import the model file
    modellib = importlib.import_module(model_filename)

    # Instantiate the model, which is a subclass of BaseModel
    model = None
    # Define the target model name
    target_model_name = model_name.replace('_', '') + 'model'

    # Find the model class name
    for name, cls in modellib.__dict__.items():
        # If the name of the class is the same as the target model name and the class is a subclass of BaseModel
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            # Set the model to the class
            model = cls

    # If the model class is not found
    if model is None:
        # Print an error message
        print("In {file}.py, there should be a subclass of BaseModel with class name that matches {target} in \
               lowercase".format(file=model_filename, target=target_model_name))
        # Exit the program
        sys.exit(0)

    # Return the model class
    return model

# Function to create the model given the configuration
def create_model(config):
    # Create the model given the configuration
    model = find_model_using_name(config.model)
    # Instantiate the model class
    instance = model(config)
    # Print the model name
    print("Model [{model}] was created".format(model=instance.get_name()))
    # Return the model
    return instance

# Function to get the model option setter
def get_option_setter(model_name):
    # Get the model class
    model_class = find_model_using_name(model_name)
    # Return the modify commandline options
    return model_class.modify_commandline_options

