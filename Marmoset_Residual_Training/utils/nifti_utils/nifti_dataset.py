from utils.utility_funcs import *
import nibabel as nib
import numpy as np
import torch
import glob

# Define the NiftiDataset class
class NiftiDataset(torch.utils.data.Dataset):

    # Constructor
    def __init__(self, data_path,
                 transforms=None,
                 train=False,
                 test=False):
        
        # Define the data paths
        self.data_path = data_path

        # Define the following paths that we use in the model
        # 1. DWI/FOD images (INPUTS 1)
        # 2. Injection center (INPUTS 2)
        # 3. Residuals (TARGETS)
        # Remember that the FOD, injection center, and residuals are what we train with
        
        # Get all the nii.gz files
        nii_gz_files = glob_files(self.data_path, "nii.gz")

        # Filter out the B0 images
        self.b0_images = [file for file in nii_gz_files if "b0" in file]
        self.b0_resized_images = [file for file in self.b0_images if "resized" in file]
        self.b0_notresized_images = [file for file in self.b0_images if "resized" not in file]

        # Filter out the residuals (TARGETS)
        self.residuals = [file for file in nii_gz_files if "subtracted" in file]
        self.residuals_flipped = [file for file in self.residuals if "flipped" in file]
        self.residuals_notflipped = [file for file in self.residuals if "unflipped" in file]

        # Get all the csv files
        csv_files = glob_files(self.data_path, "csv")

        # Filter out the injection centers
        self.injection_centers = [file for file in csv_files if "inj_center" in file]
         
        # Define the size of the lists
        self.b0_notresized_size = len(self.b0_notresized_images)
        self.residuals_notflipped_size = len(self.residuals_notflipped)

        # Sort the lists to make sure they are in the same order
        self.b0_notresized_images.sort()
        self.residuals_notflipped.sort()
        self.injection_centers.sort()

        # Define the transforms
        self.transforms = transforms

        # Define the train and test flags
        self.train = train
        self.test = test


    # Function to read an image
    def read_image(self, image_path):
        
        # Read the image using nibabel
        image = nib.load(image_path)

        # Get the image data
        image = image.get_fdata()

        # Return the image
        return image
    
    # Function to get item
    def __getitem__(self, index):

        # Get the b0 image path
        b0_image_path = self.b0_notresized_images[index]

        # Get the residual path
        residual_path = self.residuals_notflipped[index]

        # Get the injection center path
        injection_center_path = self.injection_centers[index]

        # Read the b0 image
        b0_image_array = self.read_image(b0_image_path)

        # Read the residual image
        residual_array = self.read_image(residual_path)

        # Load the injection center into a numpy array
        injection_center = np.loadtxt(injection_center_path, delimiter=',')

        # Define a dictionary to store the images
        sample = {'b0' : b0_image_array, 'residual': residual_array, 'injection_center': injection_center}

        # Return the nps. This is the final output to feed the network
        return sample["b0"], sample["residual"], sample["injection_center"]
    
    def __len__(self):
        return len(self.b0_images)


# Function to glob files
def glob_files(PATH_NAME, file_format):
    INPUT_FILES = []
    for file in glob.glob(os.path.join(PATH_NAME, os.path.join("**", "*.{}".format(file_format))), recursive=True):
        INPUT_FILES.append(file)
    return INPUT_FILES
