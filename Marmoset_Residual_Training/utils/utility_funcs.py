"""
Functions for general use
"""

import os
import torch
import regex as re
from collections import OrderedDict
import json
import glob
import nibabel as nib

from torchvision import transforms

from utils.dataloader_utils import *

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

# Function to do numerical sorting
def numerical_sort(value):
        
    # Split the value into numbers and non-numbers
    parts = re.compile(r'(\d+)').split(value)

    # Map each part as an integer if it's a number
    parts[1::2] = map(int, parts[1::2])

    # Return the parts
    return parts

# Function to list the files in a directory
def list_files(path, sort=True):
    
    # Create an empty list where the raw images are stored
    images_list = []

    # For each file in the directory
    for directory_name, subdirectory_list, file_list in os.walk(path):
        # For each file in the file list
        for file_name in file_list:
            # If the file is a nifti file
            if file_name.lower().endswith('.nii.gz') or file_name.lower().endswith('.nii'):
                # Add the file to the list
                images_list.append(os.path.join(directory_name, file_name))
            
    # If the list should be sorted
    if sort:
        # Sort the list
        images_list.sort(key=numerical_sort)

    # Return the list
    return images_list

# Function to load a json file
def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)
    
# Function to dump a json file
def dump_json(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f, indent=4, sort_keys=True)

# Function to get the newest checkpoint
def get_newest_checkpoint(checkpoint_dir):

    # List all the files in the checkpoint directory
    files = os.listdir(checkpoint_dir)

    # Get the paths to the files
    paths = [os.path.join(checkpoint_dir, basename) for basename in files]

    # Return the newest checkpoint
    return max(paths, key=os.path.getctime)

# Function to duplicate the data
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self):
        self.transform = None

    def __call__(self, x):
        return [x, x]
    
# Function to get the header from a trk/tck file
def get_streamline_header(main_data_path, tck_type):

    # Glob streamline files, depending on tck_type
    if tck_type == "tck":
        streamline_files = glob_files(main_data_path, "tck")
    elif tck_type == "trk":
        streamline_files = glob_files(main_data_path, "trk")
    else:
        raise ValueError("tck_type must be either tck or trk")
    
    # Get the tracer and non sharp files
    tracer_files = [file for file in streamline_files if "tracer" in file and "sharp" not in file]

    # Load the tractogram from the first path
    tractogram = nib.streamlines.load(tracer_files[0])

    # Get the header
    header = tractogram.header

    # Return the header
    return header

# Function to glob files
def glob_files(PATH_NAME, file_format):
    INPUT_FILES = []
    for file in glob.glob(os.path.join(PATH_NAME, os.path.join("**", "*.{}".format(file_format))), recursive=True):
        INPUT_FILES.append(file)
        break
    return INPUT_FILES