import SimpleITK as sitk
import numpy as np
import scipy

# Crop the background of an image
class CropBackground(object):

    # Constructor
    def __init__(self, image, label):

        # Define the name of the class
        self.name = 'CropBackground'

        # Set the image and label
        self.image = image
        self.label = label

    # Normalization function
    def Normalize(image):
        """
        Normalize between 0 - 255
        """

        # Create normalization and rescale filter
        normalization_filter = sitk.NormalizeImageFilter()
        rescale_filter = sitk.RescaleIntensityImageFilter()

        # Set the output maximum and minimum
        rescale_filter.SetOutputMaximum(255)
        rescale_filter.SetOutputMinimum(0)

        # Apply the filters
        image = normalization_filter.Execute(image)
        image = rescale_filter.Execute(image)

        # Return the image
        return image
    
    # Call function
    def __call__(self):

        # Normalize the image
        self.image = self.Normalize(self.image)
        self.label = self.Normalize(self.label)

        # Define a threshold filter
        threshold_filter = sitk.BinaryThresholdImageFilter()

        # Set the threshold
        threshold_filter.SetLowerThreshold(20)
        threshold_filter.SetUpperThreshold(255)

        # Set the inside and outside values
        threshold_filter.SetInsideValue(1)
        threshold_filter.SetOutsideValue(0)

        # Create region of interest filter with a specific size
        new_size = (240, 240, 120)
        roi_filter = sitk.RegionOfInterestImageFilter()
        roi_filter.SetSize([new_size[0], new_size[1], new_size[2]])

        # Apply the filters
        image_mask = threshold_filter.Execute(self.image)
        image_mask = sitk.GetArrayFromImage(image_mask)
        image_mask = np.transpose(image_mask, (2, 1, 0))

        # Get the center of mass
        center_of_mass = scipy.ndimage.measurements.center_of_mass(image_mask)

        # Get the center of mass
        x_center = np.int(center_of_mass[0])
        y_center = np.int(center_of_mass[1])

        # Set index of roi filter
        roi_filter.SetIndex([int(x_center - (new_size[0]) / 2), int(y_center - (new_size[1]) / 2), 0])

        # Apply the roi filter
        image_crop = roi_filter.Execute(self.image)
        label_crop = roi_filter.Execute(self.label)

        # Return the image and label
        return image_crop, label_crop

