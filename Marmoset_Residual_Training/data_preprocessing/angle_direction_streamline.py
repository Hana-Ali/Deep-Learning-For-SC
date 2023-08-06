import sys
from py_helpers.general_helpers import *
import numpy as np

import nibabel as nib

from nibabel.affines import apply_affine

import numpy.linalg as npl

import itertools

# Function to find the angle between 2 consecutive 3D points
def find_angle(point1, point2):

    # Get the vector between the 2 points - VECTOR 1
    vector = point2 - point1

    # Define the x-axis, y-axis and z-axis
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    # Get the norm of the first vector
    vector_norm = np.linalg.norm(vector)

    # We want to find alpha, beta and gamma, which are the angles between the vector and the x, y and z axes
    # respectively. To do this, we have the equations:
    # cos(alpha) = (vector . x_axis) / (norm(vector) * norm(x_axis))
    # cos(beta) = (vector . y_axis) / (norm(vector) * norm(y_axis))
    # cos(gamma) = (vector . z_axis) / (norm(vector) * norm(z_axis))
    # We can get the angles by taking the arccos of the above equations

    # Get the alpha angle
    alpha = np.degrees(np.arccos(np.dot(vector, x_axis) / (vector_norm * np.linalg.norm(x_axis))))

    # Get the beta angle
    beta = np.degrees(np.arccos(np.dot(vector, y_axis) / (vector_norm * np.linalg.norm(y_axis))))

    # Get the gamma angle
    gamma = np.degrees(np.arccos(np.dot(vector, z_axis) / (vector_norm * np.linalg.norm(z_axis))))

    # Print out if the angles are nan
    if np.isnan(alpha) or np.isnan(beta) or np.isnan(gamma):

        # Print the angles and vector
        print("Alpha: {}, Beta: {}, Gamma: {}".format(alpha, beta, gamma))
        print("Vector: {}".format(vector))

        # Raise a value error
        raise ValueError("Angle is nan!")
    
    # Define the number of decimals to round to
    num_decimals = 1
    # Round the angles to the number of decimal places
    alpha = round(alpha, num_decimals)
    beta = round(beta, num_decimals)
    gamma = round(gamma, num_decimals)

    # Return the angles
    return alpha, beta, gamma

# Function to map consecutive points (streamline nodes) to angles
def map_points_to_angles(points):

    # Define the list of angles
    angles = []

    # For every point in the points
    for i in range(len(points)):

        # If it's the first point, then skip
        if i == 0:
            pass
        else:
            # Get the angle between the previous point and the current point
            angle = find_angle(points[i-1], points[i])

            # Append the angle to the list of angles
            angles.append(angle)

    # Return the angles
    return angles


# Function to define the bins
def define_bins():

    # Define all possible combinations of 0s, 1s and -1s
    combinations = list(itertools.product([-1, 0, 1], repeat=3))

    # Define the number of neighbours
    num_neighbours = len(combinations)

    # For each combination, we want to map it to a number between 0 and 26
    bins = {}
    for i in range(num_neighbours):
        
        # Map the combination to the number
        bins[combinations[i]] = i
    
    # Return the bins
    return bins

# Function to threshold the normalized difference
def threshold_normalized_difference(difference, normalized_difference):

    # Define the list that holds the new values
    new_values = []

    # For each value in the normalized difference
    for i in range(len(normalized_difference)):

        # If the value is greater than 0.4, then get its sign from the difference array, then append it to the new list
        if normalized_difference[i] > 0.4:
            new_values.append(np.sign(difference[i]) * 1)
        else:
            new_values.append(0)

    # Turn it into a tuple
    new_values = tuple(new_values)

    # Return the new values
    return new_values


# Function to find the direction of the streamline (classification)
def find_direction(point1, point2):

    # Get the difference between the 2 points
    difference = point2 - point1

    # Define it as an absolute value and clip it to 1
    abs_difference = np.clip(np.abs(difference), 0, 1)

    # Normalize the difference based on the absolute value
    normalized_difference = (abs_difference / np.linalg.norm(abs_difference))

    # Based on the normalized difference, we can define the direction. The way we do this is we discretize the
    # normalized difference into 26 possible bins (each is a neighbourhood of the voxel)

    # Get the bins
    bins = define_bins()

    # Threshold the normalized difference, so that we can map it to a bin
    thresholded_difference = threshold_normalized_difference(difference, normalized_difference)

    # Map the normalized difference to a bin - this is the direction
    direction = bins[thresholded_difference]          

    # Return the direction
    return direction, thresholded_difference

# Function to map consecutive points (streamline nodes) to directions
def map_points_to_directions(points):

    # Define the list of directions
    directions = []
    directions_tuples = []

    # For every point in the points
    for i in range(len(points)):

        # If it's the first point, then skip
        if i == 0:
            pass
        else:
            # Get the direction between the previous point and the current point
            direction, thresholded_difference = find_direction(points[i-1], points[i])
            
            # Append the direction to the list of directions
            directions.append(direction)
            directions_tuples.append(thresholded_difference)
    
    # Return the directions
    return directions, directions_tuples

# Function to do the streamline node extraction
def find_angle_direction_files(trk_file, tck_file, output_path):
    
    # Get the filename of the streamline
    trk_angle_filename = trk_file.split(os.sep)[-1].replace(".trk", "_angle.npy")
    trk_direction_filename = trk_file.split(os.sep)[-1].replace(".trk", "_direction.npy")
    trk_direction_tuple_filename = trk_file.split(os.sep)[-1].replace(".trk", "_direction_tuple.npy")

    # Do the same for the tck file
    tck_angle_filename = tck_file.split(os.sep)[-1].replace(".tck", "_angle.npy")
    tck_direction_filename = tck_file.split(os.sep)[-1].replace(".tck", "_direction.npy")
    tck_direction_tuple_filename = tck_file.split(os.sep)[-1].replace(".tck", "_direction_tuple.npy")

    # Get the region folder name
    region_ID = trk_file.split(os.sep)[-2]
    region_folder = os.path.join(output_path, region_ID)
    check_output_folders(region_folder, "region_folder", wipe=False)

    # Define the trk and tck folders for this region
    trk_folder = os.path.join(region_folder, "trk")
    tck_folder = os.path.join(region_folder, "tck")
    check_output_folders(trk_folder, "trk_folder", wipe=False)
    check_output_folders(tck_folder, "tck_folder", wipe=False)

    # Define the new filepath
    trk_angle_filepath = os.path.join(trk_folder, trk_angle_filename)
    trk_direction_filepath = os.path.join(trk_folder, trk_direction_filename)
    trk_direction_tuple_filepath = os.path.join(trk_folder, trk_direction_tuple_filename)    

    tck_angle_filepath = os.path.join(tck_folder, tck_angle_filename)
    tck_direction_filepath = os.path.join(tck_folder, tck_direction_filename)
    tck_direction_tuple_filepath = os.path.join(tck_folder, tck_direction_tuple_filename)

    # Read the streamline
    trk_streamlines = nib.streamlines.load(trk_file).streamlines
    tck_streamlines = nib.streamlines.load(tck_file).streamlines

    # Define a list that will store the angles
    trk_streamline_angles, tck_streamline_angles = [], []

    # For every streamline, get the angles
    for streamline in trk_streamlines:
        trk_streamline_angles.append(map_points_to_angles(streamline))

    for streamline in tck_streamlines:
        tck_streamline_angles.append(map_points_to_angles(streamline))

    # Define lists that will store the direction (one-hot encoded class) and the direction tuples
    trk_streamline_directions, trk_streamline_direction_tuples = [], []

    # For every streamline, get the directions
    for streamline in trk_streamlines:
        # Get the directions
        directions, directions_tuples = map_points_to_directions(streamline)
        # Append them to the appropriate lists
        trk_streamline_directions.append(directions)
        trk_streamline_direction_tuples.append(directions_tuples)

    # Do the same for the tck file as well
    tck_streamline_directions, tck_streamline_direction_tuples = [], []

    # For every streamline, get the directions
    for streamline in tck_streamlines:
        # Get the directions
        directions, directions_tuples = map_points_to_directions(streamline)
        # Append them to the appropriate lists
        tck_streamline_directions.append(directions)
        tck_streamline_direction_tuples.append(directions_tuples)

    # Save the angles and directions
    np.save(trk_angle_filepath, trk_streamline_angles)
    np.save(trk_direction_filepath, trk_streamline_directions)
    np.save(trk_direction_tuple_filepath, trk_streamline_direction_tuples)

    np.save(tck_angle_filepath, tck_streamline_angles)
    np.save(tck_direction_filepath, tck_streamline_directions)
    np.save(tck_direction_tuple_filepath, tck_streamline_direction_tuples)

    print("Saved new trk angles to {}".format(trk_angle_filepath))
    print("Saved new trk directions to {}".format(trk_direction_filepath))
    print("Saved new trk tuples to {}".format(trk_direction_tuple_filepath))

    print("Saved new tck angles to {}".format(tck_angle_filepath))
    print("Saved new tck directions to {}".format(tck_direction_filepath))
    print("Saved new tck tuples to {}".format(tck_direction_tuple_filepath))

# Main function
def main():
    # Define the path to the data
    hpc = int(sys.argv[1])
    if hpc:
        tck_data_path = "/rds/general/user/hsa22/ephemeral/Brain_MINDS/model_data_w_resize/resampled_streamlines_voxels"
        trk_data_path = "/rds/general/user/hsa22/ephemeral/Brain_MINDS/model_data_w_resize/trk_data_voxels"
        output_path = "/rds/general/ephemeral/user/hsa22/ephemeral/Brain_MINDS/model_data_w_resize/streamline_stats"
    else:
        tck_data_path = "/mnt/d/Brain-MINDS/model_data_w_resize/streamline_voxels/resampled_streamlines_voxels"
        trk_data_path = "/mnt/d/Brain-MINDS/model_data_w_resize/streamline_voxels/trk_data_voxels"
        output_path = "/mnt/d/Brain-MINDS/model_data_w_resize/streamline_stats"

    check_output_folders(output_path, "output_path", wipe=False)

    # Grab the trk files - should be 156 (3 types x 52 injections)
    trk = glob_files(trk_data_path, "trk")

    # Grab all the tck files
    tck = glob_files(tck_data_path, "tck")

    # Get which region to run
    if hpc:
        file_idx = int(sys.argv[2])
        find_angle_direction_files(trk[file_idx], tck[file_idx], output_path)
    else:
        for file_idx in range(len(trk)):
            find_angle_direction_files(trk[file_idx], tck[file_idx], output_path)

if __name__ == "__main__":
    main()