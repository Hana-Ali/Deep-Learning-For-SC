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
                 transforms=None,
                 train=False,
                 test=False):
        
        # Define the data paths
        self.data_path = data_path

        # Define the following paths that we use in the model
        # 1. FOD images (INPUTS)
        # 2. Streamlines (TARGETS)
        
        # Get all the nii.gz and tck files
        nii_gz_files = glob_files(self.data_path, "nii.gz")        
        tck_files = glob_files(self.data_path, "tck")

        # Filter out the WMFOD images (INPUTS 1)
        wmfod_images = [file for file in nii_gz_files if "wmfod" in file]

        # Filter out the streamlines (TARGETS)
        streamlines = [file for file in tck_files if "tracer" in file and "sharp" not in file]

        # Prepare the lists
        self.wmfod_images = []
        self.streamlines = []

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
        streamline = nib.streamlines.load(streamline_path).streamlines

        # Return the streamline list of lists of coordinates
        return streamline
    
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
        
        # Define a dictionary to store the images
        sample = {'wmfod' : wmfod_image_array,
                  'streamlines' : streamline_list}
        
        # Return the nps. This is the final output to feed the network
        return sample["wmfod"], sample["streamlines"]
    
    def __len__(self):
        return self.streamlines_size


# Function to glob files
def glob_files(PATH_NAME, file_format):
    INPUT_FILES = []
    for file in glob.glob(os.path.join(PATH_NAME, os.path.join("**", "*.{}".format(file_format))), recursive=True):
        INPUT_FILES.append(file)
    return INPUT_FILES