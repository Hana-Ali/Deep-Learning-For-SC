from utils.utility_funcs import *
import SimpleITK as sitk
import numpy as np
import torch
import glob
import nibabel as nib
import itertools

# Set the seed
np.random.seed(0)

# Define the StreamlineDataset class
class StreamlineDataset(torch.utils.data.Dataset):

    # Constructor
    def __init__(self, data_path,
                 num_streamlines=5000,
                 transforms=None,
                 train=False,
                 test=False,
                 tck_type="trk"):
        
        # Define the data paths
        self.data_path = data_path
        
        # Define the number of streamlines
        self.num_streamlines = num_streamlines
        
        # Define the tck_type
        self.tck_type = tck_type

        # Define the following paths that we use in the model
        # 1. FOD images (INPUTS)
        # 2. Streamlines (TARGETS)
        
        # Get all the nii.gz, tck, trk, and npy files
        nii_gz_files = glob_files(self.data_path, "nii.gz")        
        tck_files = glob_files(self.data_path, "tck")
        trk_files = glob_files(self.data_path, "trk")
        npy_files = glob_files(self.data_path, "npy")

        # Filter out the WMFOD images (INPUTS 1)
        wmfod_images = [file for file in nii_gz_files if "wmfod" in file]

        # Filter out the streamlines (TARGETS)
        tck_streamlines = [file for file in tck_files if "tracer" in file and "sharp" not in file]
        trk_streamlines = [file for file in trk_files if "tracer" in file and "sharp" not in file]

        # Filter out the angle, direction, and direction tuple npy files (TARGETS)
        angle_npy_files = [file for file in npy_files if "angle" in file and "tracer" in file and "sharp" not in file]
        direction_npy_files = [file for file in npy_files if "direction" in file and "tracer" in file and "sharp" not in file]
        direction_tuple_npy_files = [file for file in npy_files if "direction_tuple" in file and "tracer" in file and "sharp" not in file]

        # Filter the angles as tck or trk
        angle_npy_tck_files = [file for file in angle_npy_files if "tck" in file]
        angle_npy_trk_files = [file for file in angle_npy_files if "trk" in file]

        # Filter the directions as tck or trk
        direction_npy_tck_files = [file for file in direction_npy_files if "tck" in file]
        direction_npy_trk_files = [file for file in direction_npy_files if "trk" in file]

        # Filter the direction tuples as tck or trk
        direction_tuple_npy_tck_files = [file for file in direction_tuple_npy_files if "tck" in file]
        direction_tuple_npy_trk_files = [file for file in direction_tuple_npy_files if "trk" in file]
           
        # Prepare the lists
        self.wmfod_images = []
        self.streamlines = []
        self.angles = []
        self.directions = []
        self.direction_tuples = []

        # If the tck type is trk, then we use the trk streamlines
        if self.tck_type == "trk":
            streamlines = trk_streamlines
        else:
            streamlines = tck_streamlines

        # For every item in the streamlines
        for i in range(len(streamlines)):

            # Get the streamline path
            streamline_path = streamlines[i]

            # Get the region ID
            region_id = streamline_path.split(os.sep)[-2]

            # Get the wmfod path that corresponds to the region ID
            wmfod_path  = [file for file in wmfod_images if region_id in file]

            # Get the angle, direction and direction tuple npy files that correspond to the region ID in either tck or trk
            if self.tck_type == "trk":
                angles_npy_path = [file for file in angle_npy_trk_files if region_id in file]
                directions_npy_path = [file for file in direction_npy_trk_files if region_id in file]
                direction_tuples_npy_path = [file for file in direction_tuple_npy_trk_files if region_id in file]
            else:
                angles_npy_path = [file for file in angle_npy_tck_files if region_id in file]
                directions_npy_path = [file for file in direction_npy_tck_files if region_id in file]
                direction_tuples_npy_path = [file for file in direction_tuple_npy_tck_files if region_id in file]

            # Raise an error if they're empty
            if angles_npy_path == [] or directions_npy_path == [] or direction_tuples_npy_path == []:
                raise ValueError("Angle, direction or direction tuple npy files are empty!")

            # If wmfod is empty it's empty, choose a random wmfod image
            if wmfod_path == []:
                wmfod_path = np.random.choice(wmfod_images)
            else:
                wmfod_path = wmfod_path[0]

            # Append the wmfod image to the list
            self.wmfod_images.append(wmfod_path)

            # Append the streamline to the list
            self.streamlines.append(streamline_path)

            # Append the angle, direction and direction tuple npy files to the list
            self.angles.append(angles_npy_path[0])
            self.directions.append(directions_npy_path[0])
            self.direction_tuples.append(direction_tuples_npy_path[0])

        # Define the size of the lists
        self.wmfod_size = len(self.wmfod_images)
        self.streamlines_size = len(self.streamlines)
        self.angles_size = len(self.angles)
        self.directions_size = len(self.directions)
        self.direction_tuples_size = len(self.direction_tuples)
        # Assert that we have the same number of wmfod as streamlines
        assert self.wmfod_size == self.streamlines_size, "WMFOD and streamlines list are not the same length!"
        # Assert that we have the same number of angles, directions and direction tuples as streamlines
        assert self.angles_size == self.streamlines_size, "Angles and streamlines list are not the same length!"
        assert self.directions_size == self.streamlines_size, "Directions and streamlines list are not the same length!"
        assert self.direction_tuples_size == self.streamlines_size, "Direction tuples and streamlines list are not the same length!"

        # Define the transforms
        self.transforms = transforms

        # Define the train and test flags
        self.train = train
        self.test = test

    # Function to read an image
    def read_image(self, image_path):
        
        # Read the image using SimpleITK
        reader = sitk.ImageFileReader()
        reader.SetFileName(image_path)
        image = reader.Execute()
        
        # Normalize the image
        image_size = image.GetSize()
        for item in range(image_size[-1]):
            output = self.normalize_image(image[:,:,:,item])
            image[:,:,:,item] = sitk.Cast(output, sitk.sitkFloat32)
        
        # Get the data from the image
        image_data = np.transpose(sitk.GetArrayFromImage(image), axes=(0, 3, 2, 1))
                
        # Return the image data
        return image_data
    
    # Function to normalize an image
    def normalize_image(self, image):
        
        # Define the normalizer
        normalizer = sitk.NormalizeImageFilter()
        rescaler = sitk.RescaleIntensityImageFilter()
        
        # Set the maximum and minimum of rescaler
        rescaler.SetOutputMaximum(255)
        rescaler.SetOutputMinimum(0)
        
        # Normalize the image (mean and std)
        image = normalizer.Execute(image)
        
        # Rescale the image (0 -> 255)
        image = rescaler.Execute(image)
        
        # Return the image
        return image
    
    # Functon to choose a random number from the array provided
    def choose_random_streamlines_range(self, provided_array):

        # Create range of length of streamlines
        random_range = np.arange(len(provided_array))

        # Randomly sample self.num_streamlines indices from the range
        random_range = np.random.choice(random_range, self.num_streamlines)

        # Return the range
        return random_range

    # Function to read a streamline
    def read_streamline(self, streamline_path):

        # Read the streamline
        streamlines = nib.streamlines.load(streamline_path).streamlines

        # Get a random range of streamlines
        streamlines_range = self.choose_random_streamlines_range(streamlines)

        # Get the streamlines from the range
        streamlines = streamlines[streamlines_range]

        # Round the streamlines
        streamlines = np.round(streamlines).astype(int)

        # Return the streamline list of lists of coordinates
        return streamlines
    
    # Function to read a npy file
    def read_npy(self, npy_path):

        # Read the npy file
        npy = np.load(npy_path, allow_pickle=True)

        # Get a random range of streamlines
        streamlines_range = self.choose_random_streamlines_range(npy)

        # Get the npy files that correspond to the streamlines range
        npy = npy[streamlines_range]

        # Return the npy
        return npy
    
    # Function to get item
    def __getitem__(self, index):
        
        # Get the wmfod image path
        wmfod_image_path = self.wmfod_images[index]

        # Get the streamline path
        streamline_path = self.streamlines[index]

        # Get the angle path
        angle_path = self.angles[index]

        # Get the direction path
        direction_path = self.directions[index]

        # Get the direction tuple path
        direction_tuple_path = self.direction_tuples[index]

        # Read the wmfod image
        wmfod_image_array = self.read_image(wmfod_image_path)

        # Read the streamline
        streamline_list = self.read_streamline(streamline_path)

        # Read the angle
        angle_array = self.read_npy(angle_path)

        # Read the direction
        direction_array = self.read_npy(direction_path)

        # Read the direction tuple
        direction_tuple_array = self.read_npy(direction_tuple_path)
        
        # Define a dictionary to store the images
        sample = {'wmfod' : wmfod_image_array,
                  'streamlines' : streamline_list,
                  'angles' : angle_array,
                  'directions' : direction_array,
                  'direction_tuples' : direction_tuple_array}
        
        # Return the nps. This is the final output to feed the network
        return sample["wmfod"], sample["streamlines"], sample["angles"], sample["directions"], sample["direction_tuples"]
    
    def __len__(self):
        return self.streamlines_size

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

# Function to, given a direction and previous node, find what the corresponding next node should be
def find_next_node(direction, previous_node):

    # If the direction is a single value, then we need to get which tuple it corresponds to
    if type(direction) == int:
        # Get the bins
        bins = define_bins()
        # Get the direction tuple
        direction_tuple = list(bins.keys())[list(bins.values()).index(direction)]
    # If it's already a tuple, then we can just use it
    else:
        direction_tuple = direction

    # Get the next node
    next_node = previous_node + np.array(direction_tuple)

    # Return the next node
    return next_node

# Function to get the next node for a given a list of directions
def reconstruct_predicted_streamline(directions):

    # The first item in directions is always the starting point
    starting_point = directions[0]

    # Define the list of nodes
    nodes = [starting_point]

    # For every direction in the directions
    for i in range(1, len(directions) - 1):

        # Get the previous node
        previous_node = nodes[i-1]

        # Get the next node
        next_node = find_next_node(directions[i], previous_node)

        # Append the next node to the list of nodes
        nodes.append(next_node)

    # Return the nodes
    return nodes

# Function to reconstruct all streamlines in a list of streamlines
def reconstruct_predicted_streamlines(streamlines):

    # Define the list of reconstructed streamlines
    reconstructed_streamlines = []

    # For every streamline in the streamlines
    for streamline in streamlines:

        # Reconstruct the streamline
        reconstructed_streamline = reconstruct_predicted_streamline(streamline)

        # Append the reconstructed streamline to the list of reconstructed streamlines
        reconstructed_streamlines.append(reconstructed_streamline)

    # Return the reconstructed streamlines
    return reconstructed_streamlines
