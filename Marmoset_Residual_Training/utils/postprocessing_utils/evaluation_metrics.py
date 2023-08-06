import nibabel as nib
import numpy as np
import os
from ..utility_funcs import *
import dipy

# Function to grab trk or tck files, load them as streamlines, and return them
def get_streamlines(streamlines_path, input_type="trk"):

    # Grab all streamline files
    streamline_files = glob_files(streamlines_path, "{ext}".format(ext=input_type))

    # Ensure that streamline files aren't empty
    assert len(streamline_files) > 0, "No {ext} files found in {path}".format(ext=input_type, path=streamlines_path)

    # Create a list that will hold all the streamlines
    streamlines_list = []

    # For every streamline file
    for i, streamline_file in enumerate(streamline_files):

        # Load the streamlines
        tractogram = nib.streamlines.load(streamline_file)
        streamlines = tractogram.streamlines

        # Add the streamlines to the list
        streamlines_list.append(streamlines)

    # Turn the list into a tractogram
    streamlines_list = nib.streamlines.Tractogram(streamlines_list, affine_to_rasmm=np.eye(4))

    # Return the list of streamlines
    return streamlines_list

# Function to convert streamlines to connectivity matrices
def streamlines_to_connectivity_matrices(streamlines_path, input_type="trk"):

    # Get the streamlines from the path
    streamlines = get_streamlines(streamlines_path, input_type=input_type)

    # Ensure that streamlines aren't empty
    assert len(streamlines) > 0, "No streamlines found in {path}".format(path=streamlines_path)

    # Create a list that will hold all the connectivity matrices
    connectivity_matrices = []

    # For every streamline
    for i, streamline in enumerate(streamlines):

        # If the type is trk, then the command is as follows
        if input_type == "trk":
            pass


