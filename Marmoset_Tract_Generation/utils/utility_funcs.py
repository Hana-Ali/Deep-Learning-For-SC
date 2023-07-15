"""
Functions for general use
"""

import os
import torch
from collections import OrderedDict

# Function to check if directory exists
def check_directory(path):
    
    # If it doesn't exist
    if not os.path.exists(path):
        # Create the directory
        os.makedirs(path)

# Function to create a directory if it doesn't exist
def make_directories(paths):

    # If the path is a list of paths, make each one
    if isinstance(paths, list) and not isinstance(paths, str):
        # For each path
        for path in paths:
            # Make the directory
            check_directory(path)
    # Otherwise, make the path
    else:
        # Make the directory
        check_directory(paths)

# Function to create a new state dictionary
def create_new_state_dict(file_name):

    # Load the state dictionary
    state_dict = torch.load(file_name)
    # Create a new state dictionary
    new_state_dict = OrderedDict()

    # For each key in the state dictionary
    for k, v in state_dict.items():
        # If the key starts with 'module.'
        if k[:6] == 'module.':
            # Remove the 'module.'
            name = k[6:]
            new_state_dict[name] = v
        # Otherwise
        else:
            # Add the key and value
            new_state_dict[k] = v

    # Return the new state dictionary
    return new_state_dict