import sys
from py_helpers.general_helpers import *
import numpy as np

import nibabel as nib

# Function to find the angle between 2 consecutive 3D vectors
def find_angle(point1, point2, point3):

    # Get the vector between points 1 and 2
    vector1 = point2 - point1

    # Get the vector between points 2 and 3
    vector2 = point3 - point2

    # Get the norm of the first vector
    vector1_norm = np.linalg.norm(vector1)

    # Get the norm of the second vector
    vector2_norm = np.linalg.norm(vector2)

    # We want to find alpha, beta and gamma, which are the angles between the  vector1 and vector2
    # respectively. To do this, we have the equations:
    # cos(alpha) = (vector1 . vector2) / (norm(vector1) * norm(vector2))
    # We can get the angles by taking the arccos of the above equations

    # Get the alpha angle
    alpha = np.degrees(np.arccos(np.dot(vector1, vector2) / (vector1_norm * vector2_norm)))

    # Print out if the angles are nan
    if np.isnan(alpha):
            
            # Print the angles and vector
            print("Alpha: {}".format(alpha))
            print("Vector1: {}".format(vector1))
            print("Vector2: {}".format(vector2))
    
            # Raise a value error
            raise ValueError("Angle is nan!")
    
    # Round to an integer
    alpha = int(round(alpha))

    # Return the angle, vector1 and vector2
    return alpha, vector1, vector2

# Function to map consecutive points (streamline nodes) to angles
def map_points_to_angles(points):

    # Define the list of angles
    angles = []

    # Define the list of points / vectors
    vectors = []
    points = []

    # For every point in the points
    for i in range(len(points)):

        # If it's the first VECTOR, then skip
        if i == 0 or i == 1:
            pass
        else:
            # Get the angle between the previous point and the current point
            angle, vector1, vector2 = find_angle(points[i-2], points[i-1], points[i])

            # Append the angle to the list of angles
            angles.append(angle)

            # Append the vectors to the list of vectors
            vectors.append([vector1, vector2])

            # Append the points to the list of points
            points.append([points[i-2], points[i-1], points[i]])

    # Return the angles
    return angles, vectors, points

# Function to do the streamline node extraction
def find_angle_direction_files(trk_file, tck_file, output_path):
    
    # Get the filename of the streamline
    trk_angle_filename = trk_file.split(os.sep)[-1].replace(".trk", "_angle_deviation.npy")
    trk_vector_filename = trk_file.split(os.sep)[-1].replace(".trk", "_vectors.npy")
    trk_points_filename = trk_file.split(os.sep)[-1].replace(".trk", "_points.npy")

    # Do the same for the tck file
    tck_angle_filename = tck_file.split(os.sep)[-1].replace(".tck", "_angle_deviation.npy")
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
    trk_angle_filepath = os.path.join(trk_folder, trk_angle_filename)
    trk_vector_filepath = os.path.join(trk_folder, trk_vector_filename)
    trk_points_filepath = os.path.join(trk_folder, trk_points_filename)

    tck_angle_filepath = os.path.join(tck_folder, tck_angle_filename)
    tck_vector_filepath = os.path.join(tck_folder, tck_vector_filename)
    tck_points_filepath = os.path.join(tck_folder, tck_points_filename)

    # Read the streamline
    trk_streamlines = nib.streamlines.load(trk_file).streamlines
    tck_streamlines = nib.streamlines.load(tck_file).streamlines

    # Define a list that will store the angles
    trk_streamline_angles, tck_streamline_angles = [], []

    # Define a list that will store the vectors
    trk_streamline_vectors, tck_streamline_vectors = [], []

    # Define a list that will store the points
    trk_streamline_points, tck_streamline_points = [], []

    # For every streamline, get the angles
    for streamline in trk_streamlines:
        angles, vectors, points = map_points_to_angles(streamline)
        trk_streamline_angles.append(angles)
        trk_streamline_vectors.append(vectors)
        trk_streamline_points.append(points)

    for streamline in tck_streamlines:
        tck_streamline_angles.append(map_points_to_angles(streamline))
        tck_streamline_vectors.append(vectors)
        tck_streamline_points.append(points)

    # Save the angles and vectors and points
    np.save(trk_angle_filepath, trk_streamline_angles)
    np.save(trk_vector_filepath, trk_streamline_vectors)
    np.save(trk_points_filepath, trk_streamline_points)

    np.save(tck_angle_filepath, tck_streamline_angles)
    np.save(tck_vector_filepath, tck_streamline_vectors)
    np.save(tck_points_filepath, tck_streamline_points)

    print("Saved new trk angles to {}".format(trk_angle_filepath))
    print("Saved new trk vectors to {}".format(trk_vector_filepath))
    print("Saved new trk points to {}".format(trk_points_filepath))

    print("Saved new tck angles to {}".format(tck_angle_filepath))
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