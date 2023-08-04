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

        # Round the streamline
        streamlines = np.round(streamlines).astype(int)

        # Create range of length of streamlines
        streamlines_range = np.arange(len(streamlines))

        # Randomly sample 9000 indices from the range
        streamlines_range = np.random.choice(streamlines_range, self.num_streamlines)

        # Get the streamlines
        streamlines = streamlines[streamlines_range]

        # Return the streamline list of lists of coordinates
        return streamlines
    
    # Function to get item
    def __getitem__(self, index):
        
        # Get the wmfod image path
        wmfod_image_path = self.wmfod_images[index]

        # Get the streamline path
        streamline_path = self.streamlines[index]

        # Read the wmfod image
        wmfod_image_array = self.read_image(wmfod_image_path)

        # Read the streamline
        streamline_list = self.read_streamline(streamline_path)

        # Get the streamline angles
        streamline_angles = []
        for streamline in streamline_list:
            streamline_angles.append(map_points_to_angles(streamline))

        # Get the streamline directions
        streamline_directions = []
        for streamline in streamline_list:
            streamline_directions.append(map_points_to_directions(streamline))
        
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

    # Define the x-axis - VECTOR 2
    x_axis = np.array([1, 0, 0])

    # Get the norm of the vector 1
    norm = np.linalg.norm(vector)

    # Get the numerator of the multiplication of the two vectors
    numerator = np.dot(vector, x_axis)

    # Get the angle between the two vectors
    angle = np.degrees(np.arccos(numerator / (norm * np.linalg.norm(x_axis))) )

    # If it's nan, print the points
    if np.isnan(angle):
        print("Point 1: ", point1)
        print("Point 2: ", point2)
        print("Vector: ", vector)
        print("Norm: ", norm)
        print("Angle: ", angle)

    # Return the angle in degrees
    return angle

# Function to map consecutive points (streamline nodes) to angles
def map_points_to_angles(points):

    # Define the list of angles
    angles = []

    print("Points: ", points)
    
    # For every point in the points
    for i in range(len(points)):

        # If it's the first point, then set the angle to 0
        if i == 0:
            angles.append(0)
        else:
            # Get the angle between the previous point and the current point
            angle = find_angle(points[i-1], points[i])

            # Append the angle to the list of angles
            angles.append(angle)

    print("Angles: ", angles)

    # Return the angles
    return angles

# Function to define the direction of the streamline (classification)
def define_direction(angles):

    # Define the direction
    direction = []

    # For every angle in the angles
    for angle in angles:

        # If the angle is between 0 and 45 degrees, then the direction is 0
        if angle >= 0 and angle <= 45:
            direction.append(0)

        # If the angle is between 45 and 90 degrees, then the direction is 1
        elif angle > 45 and angle <= 90:
            direction.append(1)

        # If the angle is between 90 and 135 degrees, then the direction is 2
        elif angle > 90 and angle <= 135:
            direction.append(2)

        # If the angle is between 135 and 180 degrees, then the direction is 3
        elif angle > 135 and angle <= 180:
            direction.append(3)

        # If the angle is between 180 and 225 degrees, then the direction is 4
        elif angle > 180 and angle <= 225:
            direction.append(4)

        # If the angle is between 225 and 270 degrees, then the direction is 5
        elif angle > 225 and angle <= 270:
            direction.append(5)

        # If the angle is between 270 and 315 degrees, then the direction is 6
        elif angle > 270 and angle <= 315:
            direction.append(6)

        # If the angle is between 315 and 360 degrees, then the direction is 7
        elif angle > 315 and angle <= 360:
            direction.append(7)

    # Return the direction
    return direction

# Function to map consecutive points (streamline nodes) to directions
def map_points_to_directions(points):

    # Get the angles
    angles = map_points_to_angles(points)

    # Get the directions
    directions = define_direction(angles)

    print("Directions: ", directions)

    # Return the directions
    return directions

