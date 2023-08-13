import sys
from py_helpers.general_helpers import *
import numpy as np

import nibabel as nib

def find_direction_points(point1, point2):

    # Get the vector between points 1 and 2
    vector1 = point2 - point1

    return vector1

# Function to find the angle between 2 consecutive 3D vectors
def find_direction(point1, point2, point3):

    # Get the vector between points 1 and 2
    vector1 = point2 - point1

    # Get the vector between points 2 and 3
    vector2 = point3 - point2

    # Subtract the vectors
    vector_difference = vector2 - vector1

    return vector_difference, vector1, vector2

# Function to map consecutive points (streamline nodes) to angles
def map_points_to_direction(points):

    # Define the list of vectors and vector differences
    vector_differences = []
    vectors = []

    # Define a list of point directions
    point_directions = []

    # Define the list of points
    new_points = []

    # For every point in the points
    for i in range(len(points)):

        # If it's the first VECTOR, then skip
        if i == 0:
            point_directions.append(find_direction_points(np.array([0, 0, 0]), points[i]))
        elif i == 1:
            point_directions.append(find_direction_points(points[i-1], points[i]))
        else:
            # Get the angle between the previous point and the current point
            vector_diff, vector1, vector2 = find_direction(points[i-2], points[i-1], points[i])

            # Get the direction of the points
            point_directions.append(find_direction_points(points[i-1], points[i]))
            
            # Append the vectors to the list of vectors
            vectors.append([vector1, vector2])

            # Append the vector difference to the list of vector differences
            vector_differences.append(vector_diff)

            # Append the points to the list of points
            new_points.append([points[i-2], points[i-1], points[i]])

    # Return the angles
    return vector_differences, point_directions, vectors, points

# Function to do the streamline node extraction
def find_angle_direction_files(trk_file, tck_file, output_path):
    
    # Get the filename of the streamline
    trk_direction_filename = trk_file.split(os.sep)[-1].replace(".trk", "_difference_direction.npy")
    trk_points_direction_filename = trk_file.split(os.sep)[-1].replace(".trk", "_points_direction.npy")
    trk_vector_filename = trk_file.split(os.sep)[-1].replace(".trk", "_vectors.npy")
    trk_points_filename = trk_file.split(os.sep)[-1].replace(".trk", "_points.npy")

    # Do the same for the tck file
    tck_direction_filename = tck_file.split(os.sep)[-1].replace(".tck", "_difference_direction.npy")
    tck_points_direction_filename = tck_file.split(os.sep)[-1].replace(".tck", "_points_direction.npy")
    tck_vector_filename = tck_file.split(os.sep)[-1].replace(".tck", "_vectors.npy")
    tck_points_filename = tck_file.split(os.sep)[-1].replace(".tck", "_points.npy")

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
    trk_direction_filepath = os.path.join(trk_folder, trk_direction_filename)
    trk_points_direction_filepath = os.path.join(trk_folder, trk_points_direction_filename)
    trk_vector_filepath = os.path.join(trk_folder, trk_vector_filename)
    trk_points_filepath = os.path.join(trk_folder, trk_points_filename)

    tck_direction_filepath = os.path.join(tck_folder, tck_direction_filename)
    tck_points_direction_filepath = os.path.join(tck_folder, tck_points_direction_filename)
    tck_vector_filepath = os.path.join(tck_folder, tck_vector_filename)
    tck_points_filepath = os.path.join(tck_folder, tck_points_filename)

    # Read the streamline
    trk_streamlines = nib.streamlines.load(trk_file).streamlines
    tck_streamlines = nib.streamlines.load(tck_file).streamlines

    # Define a list that will store the angles
    trk_streamline_directions, tck_streamline_directions = [], []

    # Define a list that will store the points directions
    trk_streamline_points_directions, tck_streamline_points_directions = [], []

    # Define a list that will store the vectors
    trk_streamline_vectors, tck_streamline_vectors = [], []

    # Define a list that will store the points
    trk_streamline_points, tck_streamline_points = [], []

    # For every streamline, get the angles
    for streamline in trk_streamlines:
        vector_differences, points_differences, vectors, points = map_points_to_direction(streamline)
        trk_streamline_directions.append(vector_differences)
        trk_streamline_points_directions.append(points_differences)
        trk_streamline_vectors.append(vectors)
        trk_streamline_points.append(points)

    for streamline in tck_streamlines:
        vector_differences, points_differences, vectors, points = map_points_to_direction(streamline)
        tck_streamline_directions.append(vector_differences)
        tck_streamline_points_directions.append(points_differences)
        tck_streamline_vectors.append(vectors)
        tck_streamline_points.append(points)

    # Save the angles and vectors and points
    np.save(trk_direction_filepath, trk_streamline_directions)
    np.save(trk_points_direction_filepath, trk_streamline_points_directions)
    np.save(trk_vector_filepath, trk_streamline_vectors)
    np.save(trk_points_filepath, trk_streamline_points)

    np.save(tck_direction_filepath, tck_streamline_directions)
    np.save(tck_points_direction_filepath, tck_streamline_points_directions)
    np.save(tck_vector_filepath, tck_streamline_vectors)
    np.save(tck_points_filepath, tck_streamline_points)

    print("Saved new trk angles to {}".format(trk_direction_filepath))
    print("Saved new trk points directions to {}".format(trk_points_direction_filepath))
    print("Saved new trk vectors to {}".format(trk_vector_filepath))
    print("Saved new trk points to {}".format(trk_points_filepath))

    print("Saved new tck angles to {}".format(tck_direction_filepath))
    print("Saved new tck points directions to {}".format(tck_points_direction_filepath))
    print("Saved new tck vectors to {}".format(tck_vector_filepath))
    print("Saved new tck points to {}".format(tck_points_filepath))

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