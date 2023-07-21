from utils.utility_funcs import *
import SimpleITK as sitk
import numpy as np
import torch

Segmentation = False

# Define the NiftiDataset class
class NiftiDataset(torch.utils.data.Dataset):

    # Constructor
    def __init__(self, data_path,
                 transforms=None,
                 train=False,
                 test=False):
        
        # Define the data paths
        self.data_path = data_path
        self.images_list = list_files(os.path.join(data_path, 'images'))
        self.labels_list = list_files(os.path.join(data_path, 'labels'))

        # Define the size of the lists
        self.images_list_size = len(self.images_list)
        self.labels_list_size = len(self.labels_list)

        # Define the transforms
        self.transforms = transforms

        # Define the train and test flags
        self.train = train
        self.test = test

        # Define the bit
        self.bit = sitk.sitkFloat32

    # Function to read an image
    def read_image(self, image_path):
        
        # Read the image
        reader = sitk.ImageFileReader()
        reader.SetFileName(image_path)
        image = reader.Execute()

        # Return the image
        return image
    
    # Function to get item
    def __getitem__(self, index):

        # Get the data path
        data_path = self.images_list[index]

        # Get the label path
        label_path = self.labels_list[index]
        
        # Read the image and label
        image = self.read_image(data_path)

        # Normalize the image
        image = self.Normalize(image)

        # Cast the image and label to a tensor
        cast_image_filter = sitk.CastImageFilter()
        cast_image_filter.SetOutputPixelType(self.bit)
        image = cast_image_filter.Execute(image)

        # If training or testing
        if self.train or self.test:
            
            # Read the label
            label = self.read_image(label_path)
            
            # Normalize the label
            label = self.Normalize(label)
            
            # Cast the label to a tensor
            cast_image_filter.SetOutputPixelType(self.bit)
            label = cast_image_filter.Execute(label)

        # If neither
        else:
            
            # Create a label image
            label = sitk.Image(image.GetSize(), self.bit)
            
            # Set the origin and spacing of the label
            label.SetOrigin(image.GetOrigin())
            label.SetSpacing(image.GetSpacing())

        # Define the sample
        sample = {'image': image, 'label': label}

        # If there is a transform
        if self.transforms:
            # For each transform
            for transform in self.transforms:
                # Apply the transform
                sample = transform(sample)

        # Convert sample to a tensor
        image_np = abs(sitk.GetArrayFromImage(sample['image']))
        label_np = abs(sitk.GetArrayFromImage(sample['label']))

        if Segmentation:
            label_np = abs(np.around(label_np))

        # To unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])  (actually it´s the contrary)
        image_np = np.transpose(image_np, (2, 1, 0))
        label_np = np.transpose(label_np, (2, 1, 0))

        label_np = (label_np - 127.5) / 127.5
        image_np = (image_np - 127.5) / 127.5

        image_np = image_np[np.newaxis, :, :, :]
        label_np = label_np[np.newaxis, :, :, :]

        # Return the nps. This is the final output to feed the network
        return torch.from_numpy(image_np), torch.from_numpy(label_np)

    # Function to normalize an image
    def Normalize(self, image):

        # Define the normalizer
        normalizer = sitk.NormalizeImageFilter()
        rescaler = sitk.RescaleIntensityImageFilter()

        # Set the maximum and minimum of rescaler
        rescaler.SetOutputMaximum(255)
        rescaler.SetOutputMinimum(0)

        # Normalize the image (mean and std)
        image = normalizer.Execute(image)
        # Rescale the image (0 to 255)
        image = rescaler.Execute(image)

        # Return the image
        return image

