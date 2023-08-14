import SimpleITK as sitk
import numpy as np
import torch
import glob
import nibabel as nib
import itertools

import torch.nn.functional as F

import PIL

import os

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
                 tck_type="trk",
                 task="classification"):
        
        # Define the data paths
        self.data_path = data_path
        
        # Define the number of streamlines
        self.num_streamlines = num_streamlines
        
        # Define the tck_type
        self.tck_type = tck_type

        # Define the task
        self.task = task

        # Define the following paths that we use in the model
        # 1. FOD images (INPUTS)
        # 2. Streamlines (TARGETS)
        
        # Get all the nii.gz, tck, trk, and npy files
        self.nii_gz_files = glob_files(self.data_path, "nii.gz")        
        self.tck_files = glob_files(self.data_path, "tck")
        self.trk_files = glob_files(self.data_path, "trk")
        self.npy_files = glob_files(self.data_path, "npy")

        # Load up the inputs
        wmfod_images, streamlines, label_npy_files = self.load_inputs()
           
        # Prepare the lists
        self.wmfod_images = []
        self.streamlines = []
        self.labels = []

        # For every item in the streamlines
        for i in range(len(streamlines)):

            # Get the streamline path
            streamline_path = streamlines[i]

            # Get the region ID
            region_id = streamline_path.split(os.sep)[-2]

            # Get the wmfod path that corresponds to the region ID
            wmfod_path  = [file for file in wmfod_images if region_id in file]

            # Get the labels that correspond to the region ID
            if self.task != "regression_coords":
                label_path = [file for file in label_npy_files if region_id in file]
            else:
                label_path = []

            # Raise an error if it's empty
            if label_path == [] and self.task != "regression_coords":
                raise ValueError("Label npy files are empty!")

            # If wmfod is empty it's empty, choose a random wmfod image
            if wmfod_path == []:
                wmfod_path = np.random.choice(wmfod_images)
            else:
                wmfod_path = wmfod_path[0]

            # Append the wmfod image to the list
            self.wmfod_images.append(wmfod_path)

            # Append the streamline to the list
            self.streamlines.append(streamline_path)

            # Append the label npy files to the list
            if self.task != "regression_coords":
                self.labels.append(label_path[0])
            
        # Define the size of the lists
        self.wmfod_size = len(self.wmfod_images)
        self.streamlines_size = len(self.streamlines)
        self.labels_size = len(self.labels)
        # Assert that we have the same number of wmfod as streamlines
        assert self.wmfod_size == self.streamlines_size, "WMFOD and streamlines list are not the same length!"
        # Assert that we have the same number of labels as streamlines (only if task isn't regression_coords, otherwise we don't need labels)
        if self.task != "regression_coords":
            assert self.labels_size == self.streamlines_size, "Labels and streamlines list are not the same length!"

        # Define the transforms
        self.transforms = transforms

        # Define the train and test flags
        self.train = train
        self.test = test

    # Function to get the inputs to the streamlines dataset (neat)
    def load_inputs(self):

        # Filter out the WMFOD images (INPUTS 1)
        wmfod_images = [file for file in self.nii_gz_files if "wmfod" in file]

        # Get the correct streamline TYPE, depending on the task and the input type
        if self.tck_type == "tck":
            streamlines = [file for file in self.tck_files if "tracer" in file and "sharp" not in file]
        elif self.tck_type == "trk":
            streamlines = [file for file in self.trk_files if "tracer" in file and "sharp" not in file]

        # Get the correct LABEL, depending on the task
        if self.task == "classification":
            label_npy_files = [file for file in self.npy_files if "direction" in file and "tracer" in file and "sharp" not in file]
            label_npy_files = self.get_tck_trk_data(label_npy_files)

        elif self.task == "regression_angles":
            label_npy_files = [file for file in self.npy_files if "angle" in file and "tracer" in file and "sharp" not in file]
            label_npy_files = self.get_tck_trk_data(label_npy_files)

        elif self.task == "regression_directions":
            label_npy_files = [file for file in self.npy_files if "direction_tuple" in file and "tracer" in file and "sharp" not in file]
            label_npy_files = self.get_tck_trk_data(label_npy_files)
        
        elif self.task == "regression_points_directions":
            label_npy_files = [file for file in self.npy_files if "points_direction" in file and "tracer" in file and "sharp" not in file]
            label_npy_files = self.get_tck_trk_data(label_npy_files)

        elif self.task == "regression_vector_directions":
            label_npy_files = [file for file in self.npy_files if "vector_direction" in file and "tracer" in file and "sharp" not in file]
            label_npy_files = self.get_tck_trk_data(label_npy_files)

        elif self.task == "regression_coords":
            label_npy_files = []
        else:
            raise ValueError("Task not recognized. Please choose from: classification, regression_angles, regression_directions, regression_coords")

        # Return the wmfods, streamlines, and labels
        return wmfod_images, streamlines, label_npy_files

    # Function to get either tck or trk data
    def get_tck_trk_data(self, data_files):

        if self.tck_type == "tck":
            data_files = [file for file in data_files if "tck" in file]
        elif self.tck_type == "trk":
            data_files = [file for file in data_files if "trk" in file]
        else:
            raise ValueError("Tck type is not valid!")
        
        return data_files

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

        # Read the tractogram
        tractogram = nib.streamlines.load(streamline_path)

        # Read the streamlines
        streamlines = tractogram.streamlines

        # Read the header
        header = tractogram.header

        # Get a random range of streamlines
        streamlines_range = self.choose_random_streamlines_range(streamlines)

        # Get the streamlines from the range
        streamlines = streamlines[streamlines_range]
        
        # Round the streamlines
        streamlines = np.round(streamlines, decimals=2)
                
        # Return the streamline list of lists of coordinates and the header
        return streamlines, header
    
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

        # Read the wmfod image
        wmfod_image_array = self.read_image(wmfod_image_path)

        # Read the streamline
        streamlines_list, header = self.read_streamline(streamline_path)

        # Choose and read label if the task is correct
        if self.task != "regression_coords":
            label_path = self.labels[index]
            label_array = self.read_npy(label_path)
        else: # Set the label to be the coordinate floats
            label_array = streamlines_list
        
        # Define a dictionary to store the images
        sample = {
                    'wmfod' : wmfod_image_array,
                    'streamlines' : streamlines_list,
                    'header' : header,
                    'labels' : label_array
                 }
         
        # Return the nps. This is the final output to feed the network
        return sample["wmfod"], sample["streamlines"], sample["labels"]
    
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
def find_next_node_classification(direction, previous_node):
    
    # Get the bins
    bins = define_bins()
    
    # Apply softmax to the direction
    direction = F.softmax(direction, dim=1)
    
    # Get the index of the maximum value along each row
    direction = torch.argmax(direction, dim=1)

    # Convert the direction and previous node to a numpy array
    direction = direction.cpu().detach().numpy()
    previous_node = previous_node.cpu().detach().numpy()

    # Create a list to hold the next nodes
    next_nodes = []

    # For each item in direction (as it's for each batch)
    for idx in range(len(direction)):

        # If the direction is a single value, then we need to get which tuple it corresponds to
        if type(direction[idx]) == np.int64:
            # Get the bins
            bins = define_bins()
            # Get the direction tuple
            direction_tuple = list(bins.keys())[list(bins.values()).index(direction[idx])]
        # If it's already a tuple, then we can just use it
        else:
            direction_tuple = direction[idx]
            
        # print("Previous node is", previous_node[idx])
        # print("Direction is", direction[idx])
        # print("Direction tuple is", direction_tuple)

        # Get the next node
        next_node = previous_node[idx] + np.array(direction_tuple)

        # Append the next node to the list of next nodes
        next_nodes.append(next_node)

    # Convert the next nodes to a numpy array of shape (batch_size, 3)
    next_nodes = np.array(next_nodes)
    next_nodes = np.reshape(next_nodes, (next_nodes.shape[0], 3))
    
    # Return the next nodes as a numpy array
    return next_nodes

# Function to get the next node for when we regress the directions
def find_next_node_points_direction(direction, previous_node):

    # Convert the previous node and direction to a numpy array
    direction = direction.cpu().detach().numpy()
    previous_node = previous_node.cpu().detach().numpy()

    # Since the points direction is just the difference between the previous node and the next node, we can just
    # add the direction to the previous node to get the next node
    next_node = previous_node + direction

    # Return the next node
    return next_node

# Function to get the angular error between the points direction and the actual direction
def get_angular_error_points_direction(points_direction, actual_direction):

    # Convert the points direction and actual direction to a numpy array
    points_direction = points_direction.cpu().detach().numpy()
    actual_direction = actual_direction.cpu().detach().numpy()

    # Get the angular error between the points direction and the actual direction
    angular_error = np.arccos(np.dot(points_direction, actual_direction) / (np.linalg.norm(points_direction) * np.linalg.norm(actual_direction)))

    # Return the angular error
    return angular_error

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
        next_node = find_next_node_classification(directions[i], previous_node)

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

# Function to glob files
def glob_files(PATH_NAME, file_format):
    INPUT_FILES = []
    for file in glob.glob(os.path.join(PATH_NAME, os.path.join("**", "*.{}".format(file_format))), recursive=True):
        INPUT_FILES.append(file)
    return INPUT_FILES