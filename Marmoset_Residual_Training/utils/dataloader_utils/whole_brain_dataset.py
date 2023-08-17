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

# Define the WholeBrainDataset class
class WholeBrainDataset(torch.utils.data.Dataset):

    # Constructor
    def __init__(self, data_path,
                 tck_type="trk",
                 task="classification"):
        
        # Define the data paths
        self.data_path = data_path
                
        # Define the tck_type
        self.tck_type = tck_type

        # Define the task
        self.task = task

        # Define the following paths that we use in the model
        # 1. FOD images (INPUTS)
        # 2. Streamlines (TARGETS)
        
        # Get all the nii.gz, tck, trk, and npy files
        self.nii_gz_files = glob_files(self.data_path, "nii.gz")
        if self.tck_type == "tck":
            self.streamline_files = glob_files(self.data_path, "tck")
        elif self.tck_type == "trk":
            self.streamline_files = glob_files(self.data_path, "trk")
        else:
            raise ValueError("Tck type is not valid!")
        self.npy_files = glob_files(self.data_path, "npy")

        # Load up the inputs
        wmfod_images, dMRI_images, streamlines, label_npy_files = self.load_inputs()
           
        # Prepare the lists
        self.wmfod_images = []
        self.dMRI_images = []
        self.streamlines = []
        self.labels = []

        # For every item in the streamlines
        for i in range(len(streamlines)):

            # Get the streamline path
            streamline_path = streamlines[i]

            # Get the region ID
            region_id = streamline_path.split(os.sep)[-2]

            # Get the wmfod path that corresponds to the region ID
            wmfod_path = [file for file in wmfod_images if region_id in file]

            # Get the dMRI path that corresponds to the region ID
            dMRI_path = [file for file in dMRI_images if region_id in file]

            # Get the labels that correspond to the region ID
            if self.task != "regression_coords" and self.task != "autoencoder":
                label_path = [file for file in label_npy_files if region_id in file]
            else:
                label_path = []

            # Raise an error if it's empty
            if label_path == [] and self.task != "regression_coords" and self.task != "autoencoder":
                raise ValueError("Label npy files are empty!")

            # If wmfod is empty it's empty, choose a random wmfod image
            if wmfod_path == []:
                wmfod_path = np.random.choice(wmfod_images)
            else:
                wmfod_path = wmfod_path[0]

            # If dMRI is empty it's empty, choose a random dMRI image
            if dMRI_path == []:
                dMRI_path = np.random.choice(dMRI_images)
            else:
                dMRI_path = dMRI_path[0]

            # Append the wmfod image to the list
            self.wmfod_images.append(wmfod_path)

            # Append the dMRI image to the list
            self.dMRI_images.append(dMRI_path)

            # Append the streamline to the list
            self.streamlines.append(streamline_path)

            # Append the label npy files to the list
            if self.task != "regression_coords" and self.task != "autoencoder":
                self.labels.append(label_path[0])
            
        # Define the size of the lists
        self.wmfod_size = len(self.wmfod_images)
        self.dMRI_size = len(self.dMRI_images)
        self.streamlines_size = len(self.streamlines)
        self.labels_size = len(self.labels)
        # Assert that we have the same number of dMRI as streamlines
        assert self.dMRI_size == self.streamlines_size, "dMRI and streamlines list are not the same length!"
        # Assert that we have the same number of wmfod as streamlines
        assert self.wmfod_size == self.streamlines_size, "WMFOD and streamlines list are not the same length!"
        # Assert that we have the same number of labels as streamlines (only if task isn't regression_coords, otherwise we don't need labels)
        if self.task != "regression_coords" and self.task != "autoencoder":
            assert self.labels_size == self.streamlines_size, "Labels and streamlines list are not the same length!"

    # Function to get the inputs to the streamlines dataset (neat)
    def load_inputs(self):

        # Filter out the WMFOD images (INPUTS 1)
        wmfod_images = [file for file in self.nii_gz_files if "wmfod" in file]

        # Filter out the dMRI images (INPUTS 2)
        dMRI_images = [file for file in self.nii_gz_files if "dMRI" in file]

        # Get the correct streamline TYPE, depending on the task and the input type
        streamlines = [file for file in self.streamline_files if "tracer" in file and "sharp" not in file]

        # Get the correct LABEL, depending on the task
        if self.task == "classification":
            label_npy_files = [file for file in self.npy_files if "direction" in file and "points" not in file and "tracer" in file and "sharp" not in file]
            label_npy_files = self.get_tck_trk_data(label_npy_files)

        elif self.task == "regression_angles":
            label_npy_files = [file for file in self.npy_files if "angle" in file and "tracer" in file and "sharp" not in file]
            label_npy_files = self.get_tck_trk_data(label_npy_files)

        elif self.task == "regression_directions":
            label_npy_files = [file for file in self.npy_files if "direction_tuple" in file and "tracer" in file and "sharp" not in file]
            label_npy_files = self.get_tck_trk_data(label_npy_files)
        
        elif self.task == "regression_points_directions":
            label_npy_files = [file for file in self.npy_files if "points_direction_no_first" in file and "tracer" in file and "sharp" not in file]
            label_npy_files = self.get_tck_trk_data(label_npy_files)

        elif self.task == "regression_vector_directions":
            label_npy_files = [file for file in self.npy_files if "vector_direction" in file and "tracer" in file and "sharp" not in file]
            label_npy_files = self.get_tck_trk_data(label_npy_files)

        elif self.task == "regression_coords" or self.task == "autoencoder":
            label_npy_files = []
        else:
            raise ValueError("Task not recognized. Please choose from: classification, regression_angles, regression_directions, regression_coords, autoencoder")

        # Return the wmfods, streamlines, and labels
        return wmfod_images, dMRI_images, streamlines, label_npy_files

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
    
    # Function to read a streamline
    def read_streamline(self, streamline_path):

        # Read the tractogram
        tractogram = nib.streamlines.load(streamline_path)

        # Read the streamlines
        streamlines = tractogram.streamlines

        # Read the header
        header = tractogram.header
        
        # Round the streamlines
        streamlines = np.round(streamlines, decimals=2)
                
        # Return the streamline list of lists of coordinates and the header
        return streamlines, header
    
    # Function to read a npy file
    def read_npy(self, npy_path):

        # Read the npy file
        npy = np.load(npy_path, allow_pickle=True)

        # Return the npy
        return npy
    
    # Function to get the affine of a dMRI image
    def get_affine(self, dMRI_path):

        # Load the dMRI image
        dMRI_image = nib.load(dMRI_path)

        # Get the affine
        affine = dMRI_image.affine

        # Return the affine
        return affine
    
    # Function to get item
    def get_brain_data(self):

        # Get a random index between 0 and the size of the dataset
        index = np.random.randint(0, self.streamlines_size)
        
        # Get the wmfod image path
        wmfod_image_path = self.wmfod_images[index]

        # Get the dMRI image path
        dMRI_image_path = self.dMRI_images[index]

        # Get the streamline path
        streamline_path = self.streamlines[index]

        # Get the name of the brain we're using
        brain_name = streamline_path.split(os.sep)[-2]

        # Read the wmfod image
        wmfod_image_array = self.read_image(wmfod_image_path)

        # Get the DWI affine
        affine = self.get_affine(dMRI_image_path)

        # Read the streamline
        streamlines_list, header = self.read_streamline(streamline_path)

        # Choose and read label if the task is correct
        if self.task != "regression_coords" and self.task != "autoencoder":
            label_path = self.labels[index]
            label_array = self.read_npy(label_path)
        elif self.task == "autoencoder":
            label_array = np.zeros((2, 2))
        else: # Set the label to be the coordinate floats
            label_array = streamlines_list
                                
        # Define a dictionary to store the images
        brain_data = {
                        'wmfod' : wmfod_image_array,
                        'streamlines' : streamlines_list,
                        'header' : header,
                        'affine' : affine,
                        'labels' : label_array
                    }
         
        # Return the nps. This is the final output to feed the network
        return brain_data, brain_name 
    
    def get_dataset_length(self):
        return self.streamlines_size

# Function to glob files
def glob_files(PATH_NAME, file_format):
    INPUT_FILES = []
    for file in glob.glob(os.path.join(PATH_NAME, os.path.join("**", "*.{}".format(file_format))), recursive=True):
        INPUT_FILES.append(file)
    return INPUT_FILES