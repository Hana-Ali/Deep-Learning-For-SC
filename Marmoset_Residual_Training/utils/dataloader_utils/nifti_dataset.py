from utils.utility_funcs import *
import SimpleITK as sitk
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
        
        # Get all the nii.gz and csv files
        nii_gz_files = glob_files(self.data_path, "nii.gz")        
        csv_files = glob_files(self.data_path, "csv")

        # Filter out the B0 images (INPUTS 1)
        b0_images = [file for file in nii_gz_files if "b0" in file and "resized" in file]
        
        # Filter out the WMFOD images (INPUTS 2)
        wmfod_images = [file for file in nii_gz_files if "wmfod" in file]

        # Filter out the injection centers (INPUTS 2)
        injection_centers = [file for file in csv_files if "inj_center" in file]

        # Filter out the residuals (TARGETS)
        residuals = [file for file in nii_gz_files if "subtracted" in file and "resized" in file]

        # Prepare the lists
        self.b0_images = []
        self.wmfod_images = []
        self.residuals = []
        self.injection_centers = []

        # For every item in the residuals
        for i in range(len(residuals)):

            # Get the residual path
            residual_path = residuals[i]

            # Get the region ID
            region_id = residual_path.split(os.sep)[-2]

            # Get whether it's flipped or not
            is_not_flipped = "unflipped" in residual_path

            # Get the b0 path that corresponds to the region ID
            b0_path = [file for file in b0_images if region_id in file]
            
            # Get the wmfod path that corresponds to the region ID
            wmfod_path  = [file for file in wmfod_images if region_id in file]

            # Get the injection center that corresponds to the region ID
            injection_center_path = [file for file in injection_centers if region_id in file][0]

            # If it's empty, choose a random b0 image
            if b0_path == []:
                b0_path = np.random.choice(b0_images)
            else:
                b0_path = b0_path[0]
                
            # Same for wmfod
            if wmfod_path == []:
                wmfod_path = np.random.choice(wmfod_images)
            else:
                wmfod_path = wmfod_path[0]

            # Append the B0 image to the list
            self.b0_images.append(b0_path)
            
            # Append the wmfod image to the list
            self.wmfod_images.append(wmfod_path)

            # Append the residual to the list, and add the is_flipped flag
            self.residuals.append((residual_path, not is_not_flipped))

            # Append the injection center to the list
            self.injection_centers.append(injection_center_path)

        # Define the size of the lists
        self.b0_size = len(self.b0_images)
        self.wmfod_size = len(self.wmfod_images)
        self.residuals_size = len(self.residuals)
        self.injection_centers_size = len(self.injection_centers)
        
        # Assert that we have the same number of b0 as residuals
        assert self.b0_size == self.residuals_size, "B0 and residuals list are not the same length!"
        assert self.wmfod_size == self.b0_size, "WMFOD and B0 list are not the same length!"

        # Define the transforms
        self.transforms = transforms

        # Define the train and test flags
        self.train = train
        self.test = test


    # Function to read an image
    def read_image(self, image_path, wmfod=False):
        
        # Read the image using SimpleITK
        reader = sitk.ImageFileReader()
        reader.SetFileName(image_path)
        image = reader.Execute()
        
        # Normalize the image
        if wmfod:
            image_size = image.GetSize()
            for item in range(image_size[-1]):
                output = self.normalize_image(image[:,:,:,item])
                image[:,:,:,item] = sitk.Cast(output, sitk.sitkFloat32)
        else:
            image = self.normalize_image(image)
        
        # Get the data from the image
        if wmfod:
            image_data = np.transpose(sitk.GetArrayFromImage(image), axes=(0, 3, 2, 1))
        else:
            image_data = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))
        
        # Expand the dimension of the image (only if it's not wmfod)
        if not wmfod:
            image_data = np.expand_dims(image_data, axis=0)
        
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
    
    # Function to get item
    def __getitem__(self, index):

        # Get the b0 image path
        b0_image_path = self.b0_images[index]
        
        # Get the wmfod image path
        wmfod_image_path = self.wmfod_images[index]

        # Get the residual path and the is_flipped flag
        residual_path = self.residuals[index][0]
        is_flipped = self.residuals[index][1]

        # Get the injection center path
        injection_center_path = self.injection_centers[index]

        # Read the b0 image
        b0_image_array = self.read_image(b0_image_path)
        
        # Read the wmfod image
        wmfod_image_array = self.read_image(wmfod_image_path, wmfod=True)
        
        # Read the residual image
        residual_array = self.read_image(residual_path)

        # Load the injection center into a numpy array
        injection_center = np.loadtxt(injection_center_path, delimiter=',')
        
        # Define a dictionary to store the images
        sample = {'b0' : b0_image_array, 
                  'wmfod' : wmfod_image_array,
                  'residual': (residual_array, is_flipped), 
                  'injection_center': injection_center}

        # Return the nps. This is the final output to feed the network
        return sample["b0"], sample["wmfod"], sample["residual"], sample["injection_center"]
    
    def __len__(self):
        return self.b0_size