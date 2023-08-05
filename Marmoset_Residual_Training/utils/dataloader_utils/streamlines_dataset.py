from utils.utility_funcs import *
import SimpleITK as sitk
import numpy as np
import torch
import glob
import nibabel as nib

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
        
        # Get all the nii.gz, tck and trk files
        nii_gz_files = glob_files(self.data_path, "nii.gz")        
        tck_files = glob_files(self.data_path, "tck")
        trk_files = glob_files(self.data_path, "trk")

        # Filter out the WMFOD images (INPUTS 1)
        wmfod_images = [file for file in nii_gz_files if "wmfod" in file]

        # Filter out the streamlines (TARGETS)
        tck_streamlines = [file for file in tck_files if "tracer" in file and "sharp" not in file]
        trk_streamlines = [file for file in trk_files if "tracer" in file and "sharp" not in file]

        # Prepare the lists
        self.wmfod_images = []
        self.streamlines = []

        # If the tck type is trk, then we use the trk streamlines
        if tck_type == "trk":
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

            # If it's empty, choose a random wmfod image
            if wmfod_path == []:
                wmfod_path = np.random.choice(wmfod_images)
            else:
                wmfod_path = wmfod_path[0]

            # Append the wmfod image to the list
            self.wmfod_images.append(wmfod_path)

            # Append the streamline to the list
            self.streamlines.append(streamline_path)

        # Define the size of the lists
        self.wmfod_size = len(self.wmfod_images)
        self.streamlines_size = len(self.streamlines)

        # Assert that we have the same number of wmfod as streamlines
        assert self.wmfod_size == self.streamlines_size, "WMFOD and streamlines list are not the same length!"

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
    
    # Function to read a streamline
    def read_streamline(self, streamline_path):

        # Read the streamline
        streamlines = nib.streamlines.load(streamline_path).streamlines

        # Create range of length of streamlines
        streamlines_range = np.arange(len(streamlines))

        # Randomly sample self.num_streamlines indices from the range
        streamlines_range = np.random.choice(streamlines_range, self.num_streamlines)

        # Get the streamlines
        streamlines = streamlines[streamlines_range]
        
        # Get the streamline angles
        streamline_angles = []
        for streamline in streamlines:
            streamline_angles.append(map_points_to_angles(streamline))

        # Get the streamline directions
        streamline_directions = []
        for streamline in streamlines:
            streamline_directions.append(map_points_to_directions(streamline))

        return

        # Round the streamlines
        streamlines = np.round(streamlines).astype(int)

        # Return the streamline list of lists of coordinates
        return (streamlines, streamline_angles, streamline_directions)
    
    # Function to get item
    def __getitem__(self, index):
        
        # Get the wmfod image path
        wmfod_image_path = self.wmfod_images[index]

        # Get the streamline path
        streamline_path = self.streamlines[index]

        # Read the wmfod image
        wmfod_image_array = self.read_image(wmfod_image_path)

        # Read the streamline
        (streamline_list, streamline_angles, 
         streamline_directions) = self.read_streamline(streamline_path)
        
        # Define a dictionary to store the images
        sample = {'wmfod' : wmfod_image_array,
                  'streamlines' : streamline_list,
                  'angles' : np.array(streamline_angles),
                  'directions' : np.array(streamline_directions)}
        
        # Return the nps. This is the final output to feed the network
        return sample["wmfod"], sample["streamlines"], sample["angles"], sample["directions"]
    
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

# Function to define the direction of the streamline (classification)
def find_direction(point1, point2):

    # Get the difference between the 2 points
    difference = point2 - point1

    # Print the difference
    print("Difference: {}".format(difference))

    # Return the direction
    return difference

# Function to map consecutive points (streamline nodes) to directions
def map_points_to_directions(points):

    # Define the list of directions
    directions = []

    # For every point in the points
    for i in range(len(points)):

        # If it's the first point, then skip
        if i == 0:
            pass
        else:
            # Get the direction between the previous point and the current point
            direction = find_direction(points[i-1], points[i])

            # Append the direction to the list of directions
            directions.append(direction)

    # Return the directions
    return directions

# Function to, given a direction (discretized angle), find the vector that corresponds to that direction
def find_vector(angle):

    # Define the x-axis, y-axis and z-axis
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    # Define the vector
