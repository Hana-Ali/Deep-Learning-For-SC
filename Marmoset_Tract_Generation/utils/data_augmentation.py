"""
This file is dedicated to data augmentation.
Specifically, we want to be able to augment the DWI data in a way that is consistent with the
augmentation of the corresponding tractogram.

The augmentation methods are:
    - Random rotation
    - Random translation
    - Random scaling
    - Random flipping
    - Random noise
    - Random cropping

The augmentation methods are implemented as functions that take in a DWI image and a tractogram and
return the augmented DWI image and tractogram.
"""

import numpy as np
import SimpleITK as sitk
import scipy

interpolator_image = sitk.sitkLinear                 # interpolator image
interpolator_label = sitk.sitkLinear                  # interpolator label

_interpolator_image = 'linear'          # interpolator image
_interpolator_label = 'linear'          # interpolator label

Segmentation = False

# Resample the image to a given voxel size
class Resample(object):
    """
    Resampling the volume in a sample to a given voxel size
    
    Args:
        voxel_size (float or tuple) - The voxel size to resample to
        -> Float - isotropic resampling
        -> Tuple - anisotropic resampling
        This function only does linear interpolation
    """

    # Constructor
    def __init__(self, new_voxel_size, check_voxel_size=False):

        # Define the name of the class
        self.name = 'Resample'

        # Check what the new resolution is
        if isinstance(new_voxel_size, float):
            # Set the new voxel size
            self.new_voxel_size = (new_voxel_size, new_voxel_size, new_voxel_size)
        else:
            # Set the new voxel size
            self.new_voxel_size = new_voxel_size
        
        self.check_voxel_size = check_voxel_size

    # Call function
    def __call__(self, sample):

        # Get the image and the tractogram
        image, tractogram = sample['image'], sample['tractogram']

        # If check is true, then actually resample
        if self.check_voxel_size:
            # Resample the image and label
            image = self.resample_sitk_image(image, spacing=self.new_voxel_size, interpolator=_interpolator_image)
            label = self.resample_sitk_image(label, spacing=self.new_voxel_size, interpolator=_interpolator_label)

            # Return the resampled image and label
            return {'image': image, 'tractogram': tractogram}
        
        # If not, return the original image and tractogram
        if not self.check_voxel_size:
            # Return the original image and tractogram
            return {'image': image, 'tractogram': tractogram}

    # Resample a sitk image
    def resample_sitk_image(self, image, spacing=None, interpolator=None, fill_value=0):
        """
        Resample a sitk image to a given voxel size, or new grid.
        -> No spacing - isotropically to the smallest value in current spacing (in-plane rez)
        -> No interpolation - derived from the original image
        -> Binary inputs (e.g. masks) - use nearest neighbour interpolation (or linear)

        Parameters
        ----------

        image : sitk.Image
            The image to resample
        spacing : list or tuple, optional
            The new voxel size to resample to
        interpolator : sitk interpolator, optional
            The interpolator to use
        fill_value : int, optional

        Returns
        -------
        sitk.Image
            The resampled image
        """

        # Define valid interpolators
        SITK_INTERPOLATOR_DICT = {
            'nearest': sitk.sitkNearestNeighbor,
            'linear': sitk.sitkLinear,
            'gaussian': sitk.sitkGaussian,
            'label_gaussian': sitk.sitkLabelGaussian,
            'bspline': sitk.sitkBSpline,
            'hamming_sinc': sitk.sitkHammingWindowedSinc,
            'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
            'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
            'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
        }

        # Check if the image is a sitk image
        if not isinstance(image, sitk.Image):
            raise ValueError('Image must be a sitk image')
        
        # Check if a path to an image is given - not the image
        if isinstance(image, str):
            # Read the image
            image = sitk.ReadImage(image)

        # Check if the interpolation is given
        if not interpolator:
            # Set the interpolator to linear
            interpolator = 'linear'
            # Get the pixel ID value (the type of image we are working with)
            pixel_id = image.GetPixelIDValue()

            # Check that the pixel id is valid
            # if pixel_id not in [sitk.sitkUInt8, sitk.sitkUInt16, sitk.sitkFloat32]:
            if pixel_id not in [1, 2, 4]:
                raise NotImplementedError("The pixel type {} is not supported. Can only infer for 8-bit unsigned, \
                                          or 16/32-bit signed integers".format(pixel_id))
            # If 8-bit unsigned integer, then interpolate using nearest neighbour
            if pixel_id == 1:
                interpolator = 'nearest'
        # If the interpolation is given
        else:
            # Check if the interpolation is valid
            if interpolator not in SITK_INTERPOLATOR_DICT.keys():
                raise ValueError('The interpolator {} is not valid. Should be one of {}'.format(interpolator, SITK_INTERPOLATOR_DICT.keys()))
            
            # Get the interpolator
            interpolator = SITK_INTERPOLATOR_DICT[interpolator]
        
        # Get the original dimensions
        original_dimensions = image.GetDimension()
        # Get the pixel ID
        pixel_id = image.GetPixelIDValue()
        # Get the origin of the image
        original_origin = image.GetOrigin()
        # Get the direction of the image
        original_direction = image.GetDirection()
        # Get the spacing of the image
        original_spacing = np.array(image.GetSpacing())
        # Get the size of the image
        original_size = np.array(image.GetSize(), dtype=np.int)

        # Check if the spacing is given
        if not spacing:
            # Get the minimum spacing
            min_spacing = np.min(original_spacing)
            # Set the new spacing
            new_spacing = [min_spacing] * original_dimensions
        # If the spacing is given
        else:
            # Set the new spacing
            new_spacing = [float(s) for s in spacing]

        # Get the new size
        new_size = np.ceil(original_size * (original_spacing / new_spacing)).astype(np.int)
        new_size = [int(s) for s in new_size] # List not numpy array as SITK needs this

        # Get the resample filter
        resample_filter = sitk.ResampleImageFilter()

        # Get the new resampled image
        new_image = resample_filter.Execute(image, 
                                            new_size, 
                                            sitk.Transform(), 
                                            interpolator, 
                                            original_origin, 
                                            new_spacing, 
                                            original_direction, 
                                            fill_value, 
                                            pixel_id)
        
        # Return the new image
        return new_image

# Augment the image and tractogram
class Augmentation(object):
    """
    Application of transformations to the image and tractogram
    (Random noise)
    """

    # Constructor
    def __init__(self):
            
        # Define the name of the class
        self.name = 'Augmentation'

    # Call function
    def __call__(self, sample):

        # Get the image and the tractogram
        image, tractogram = sample['image'], sample['tractogram']

        # Augment the image and tractogram
        image, tractogram = self.augment_sitk_image(image, tractogram)

        # Return the augmented image and tractogram
        return {'image': image, 'tractogram': tractogram}
    
    # Augment a sitk image
    def augment_sitk_image(self, image, tractogram):

        # Choose a random number
        random_noise = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])

        # If the random number is 0 - no augmentation
        if random_noise == 0:
            # Return the image and tractogram
            return image, tractogram
        
        # If the random number is 1 - Additive Gaussian noise
        if random_noise == 1:

            # Get the mean and standard deviation
            mean = np.random.uniform(0, 1)
            std = np.random.uniform(0, 1)

            # Get a noise filter and set its mean and std
            noise_filter = sitk.AdditiveGaussianNoiseImageFilter()
            noise_filter.SetMean(mean)
            noise_filter.SetStandardDeviation(std)

            # Apply the noise filter to the image
            image = noise_filter.Execute(image)
            # If no segmentation (?)
            if Segmentation is False:
                # Apply noise to label
                label = noise_filter.Execute(label)

            # Return the image and tractogram
            return image, tractogram

        # If the random number is 2 - Recursive Gaussian noise
        if random_noise == 2:

            # Get the sigma
            sigma = np.random.uniform(0, 1.5)

            # Get a noise filter and set its sigma
            noise_filter = sitk.RecursiveGaussianImageFilter()
            noise_filter.SetOrder(0)
            noise_filter.SetSigma(sigma)

            # Apply the noise filter to the image
            image = noise_filter.Execute(image)
            # If no segmentation (?)
            if Segmentation is False:
                # Apply noise to label
                label = noise_filter.Execute(label)

            # Return the image and tractogram
            return image, tractogram
        
        # If the random number is 3 - Random rotation in x, y and z
        if random_noise == 3:

            # Get the rotation angle
            rotation_angle_x = np.random.randint(-20, 20)
            rotation_angle_y = np.random.randint(-20, 20)
            rotation_angle_z = np.random.randint(-180, 180)

            # Rotate the image and tractogram
            image = self.rotate_image(image, rotation_angle_x, rotation_angle_y, rotation_angle_z)
            tractogram = self.rotate_image(tractogram, rotation_angle_x, rotation_angle_y, rotation_angle_z)

            # Return the image and tractogram
            return image, tractogram
        
        # If the random number is 4 - BSpline Deformation
        if random_noise == 4:

            # Define the spline order
            spline_order = 3

            # Define the physical dimensions of the image
            domain_physical_dimensions = [image.GetSize()[0] * image.GetSpacing()[0],
                                            image.GetSize()[1] * image.GetSpacing()[1],
                                            image.GetSize()[2] * image.GetSpacing()[2]]
            
            # Define the BSpline transform
            bspline_transform = sitk.BSplineTransform(3, spline_order)
            # Define the domain and transform mesh size
            bspline_transform.SetTransformDomainOrigin(image.GetOrigin())
            bspline_transform.SetTransformDomainDirection(image.GetDirection())
            bspline_transform.SetTransformDomainPhysicalDimensions(domain_physical_dimensions)
            bspline_transform.SetTransformDomainMeshSize((10, 10, 10))

            # Define the displacement of control points
            randomness = 10
            original_control_point_displacements = np.random.random(len(bspline_transform.GetParameters())) * randomness
            # Set the parameters of the BSpline transform
            bspline_transform.SetParameters(original_control_point_displacements)

            # Resample the image and tractogram
            image = self.resample_image_from_ref(image, bspline_transform)
            tractogram = self.resample_image_from_ref(tractogram, bspline_transform)

            # Return the image and tractogram
            return image, tractogram
        
        # If the random number is 5 - Random flipping
        if random_noise == 5:
                
            # Get the flip axes
            flip_axes = np.random.choice([0, 1])

            # Flip the image and tractogram
            image = self.flip_image(image, flip_axes)
            tractogram = self.flip_image(tractogram, flip_axes)

            # Return the image and tractogram
            return image, tractogram
        
        # If the random number is 6 - Random brightness change
        if random_noise == 6:

            # Change the brightness of the image
            image = self.change_brightness(image)

            # If no segmentation (?)
            if Segmentation is False:
                # Change the brightness of the label
                label = self.change_brightness(label)

            # Return the image and tractogram
            return image, tractogram
        
        # If the random number is 7 - Random contrast change
        if random_noise == 7:

            # Change the contrast of the image
            image = self.change_contrast(image)

            # If no segmentation (?)
            if Segmentation is False:
                # Change the contrast of the label
                label = self.change_contrast(label)

            # Return the image and tractogram
            return image, tractogram
        
        # If the random number is 8 - Random translation
        if random_noise == 8:

            # Get the offset
            offset_1 = np.random.randint(-40, 40)
            offset_2 = np.random.randint(-40, 40)
            offset = [offset_1, offset_2]

            # Translate the image and tractogram
            image = self.translate_image(image, offset)
            tractogram = self.translate_image(tractogram, offset)

            # Return the image and tractogram
            return image, tractogram
        
        # If the random number if 9 - Random rotation in the z axis
        if random_noise == 9:

            # Get the rotation angle
            rotation_angle_z = np.random.randint(-180, 180)

            # Rotate the image and tractogram
            image = self.rotate_image(image, 0, 0, rotation_angle_z)
            tractogram = self.rotate_image(tractogram, 0, 0, rotation_angle_z)

            # Return the image and tractogram
            return image, tractogram
        
        # If the random number is 10 - Random rotation in x
        if random_noise == 10:

            # Get the rotation angle
            rotation_angle_x = np.random.randint(-20, 20)

            # Rotate the image and tractogram
            image = self.rotate_image(image, rotation_angle_x, 0, 0)
            tractogram = self.rotate_image(tractogram, rotation_angle_x, 0, 0)

            # Return the image and tractogram
            return image, tractogram
        
        # If the random number is 11 - Random rotation in y
        if random_noise == 11:

            # Get the rotation angle
            rotation_angle_y = np.random.randint(-20, 20)

            # Rotate the image and tractogram
            image = self.rotate_image(image, 0, rotation_angle_y, 0)
            tractogram = self.rotate_image(tractogram, 0, rotation_angle_y, 0)

            # Return the image and tractogram
            return image, tractogram
        
        # If the random image is 12 - Random adjustment
        if random_noise == 12:

            # Adjust the image
            image = self.adjust_image(image)

            # Return the image and tractogram
            return image, tractogram

    # Define the rotation function for the image
    def rotate_image(self, image, rotation_angle_x, rotation_angle_y, rotation_angle_z):
        """
        This function rotates the image by a given angle in x, y and z
        """

        # Get the angles in radians
        rotation_angle_x = np.radians(rotation_angle_x)
        rotation_angle_y = np.radians(rotation_angle_y)
        rotation_angle_z = np.radians(rotation_angle_z)

        # Get the center of the image
        center = self.get_center(image)

        # Get the Euler 3D transform
        euler_transform = sitk.Euler3DTransform(center, rotation_angle_x, rotation_angle_y, rotation_angle_z, (0, 0, 0))

        # Set the center and rotation
        euler_transform.SetCenter(center) 
        euler_transform.SetRotation(rotation_angle_x, rotation_angle_y, rotation_angle_z)

        # Resample the image from a reference
        resampled_image = self.resample_image_from_ref(image, euler_transform)

        # Return the resampled image
        return resampled_image

    # Get the center of the image
    def get_center(self, image):
        # Get the size of the image
        width, height, depth = image.GetSize()
        # Get the center of the image
        center = image.TransformIndexToPhysicalPoint((int(np.ceil(width / 2)),
                                                    int(np.ceil(height / 2)),
                                                    int(np.ceil(depth / 2))))
        # Return the center
        return center
    
    # Resample from a reference image
    def resample_image_from_ref(self, image, transform):
        # Get the reference image
        reference_image = image
        
        # Get the interpolator
        interpolator = interpolator_image
        
        # Get the default pixel value
        default_value = 0

        # Resample the image
        resampled_image = sitk.Resample(image, reference_image, transform, interpolator, default_value)

        # Return the resampled image
        return resampled_image
    
    # Flip the image
    def flip_image(self, image, flip_axes):

        # Get the image array
        image_array = np.transpose(sitk.GetArrayFromImage(image), axis=(2, 1, 0))

        # Get the spacing, direction and origin of the image
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        # Flip the image array, depending on axes
        if flip_axes == 0:
            image_array = np.fliplr(image_array)
        if flip_axes == 1:
            image_array = np.flipud(image_array)

        # Get the new image
        new_image = sitk.GetImageFromArray(np.transpose(image_array, (2, 1, 0)))

        # Set the new spacing, direction and origin
        new_image.SetSpacing(spacing)
        new_image.SetDirection(direction)
        new_image.SetOrigin(origin)

        # Return the new image
        return new_image
        
    # Change the brightness of the image
    def change_brightness(self, image):
        
        # Get the image array
        image_array = np.transpose(sitk.GetArrayFromImage(image), axis=(2, 1, 0))

        # Get the spacing, direction and origin of the image
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        # Define the maximuma nd minimum brightness
        max_brightness, min_brightness = 255, 0

        # Get the brightness change as a random int
        brightness_change = np.random.randint(-20, 20)

        # Change the brightness of the image
        image_array = image_array + brightness_change

        # Clip the image array
        image_array = np.clip(image_array, min_brightness, max_brightness)

        # Get the new image
        new_image = sitk.GetImageFromArray(np.transpose(image_array, axes=(2, 1, 0)))

        # Set the new spacing, direction and origin
        new_image.SetSpacing(spacing)
        new_image.SetDirection(direction)
        new_image.SetOrigin(origin)

        # Return the new image
        return new_image
    
    # Change the contrast of the image
    def change_contrast(self, image):

        # Get the image array
        image_array = np.transpose(sitk.GetArrayFromImage(image), axis=(2, 1, 0))

        # Get the spacing, direction and origin of the image
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        # Get the shape of the image array
        shape = image_array.shape

        # Get the number of total pixels
        num_pixels = shape[0] * shape[1] * shape[2]

        # Get the IOD? and luminanza?
        IOD = np.sum(image_array)
        luminanza = int(IOD / num_pixels)

        # Get the contrast change as a random int
        contrast_change = np.random.randint(-20, 20)

        # Define the array minus luminanzes
        array_minus_luminanza = image_array - luminanza
        array_minus_luminanza = array_minus_luminanza * abs(contrast_change) / 100

        # Change the image depending on the contrast
        if contrast_change >= 0:
            # Add the array minus luminanza to the image array
            new_image_array = image_array + array_minus_luminanza
            # Clip the image array
            new_image_array = np.clip(new_image_array, 0, 255)
        else:
            # Subtract the array minus luminanza from the image array
            new_image_array = image_array - array_minus_luminanza
            # Clip the image array
            new_image_array = np.clip(new_image_array, 0, 255)

        # Get the new image
        new_image = sitk.GetImageFromArray(np.transpose(new_image_array, axes=(2, 1, 0)))
        
        # Set the new spacing, direction and origin
        new_image.SetSpacing(spacing)
        new_image.SetDirection(direction)
        new_image.SetOrigin(origin)

        # Return the new image
        return new_image
    
    # Translate the image
    def translate_image(self, image, offset, isSeg=False):

        # Order depending on segmentation or not
        order = 0 if isSeg == True else 5

        # Get the image array
        image_array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))

        # Get the spacing, direction and origin of the image
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        # Shift the image with scipy
        image_array = scipy.ndimage.interpolation.shift(image_array, (int(offset[0]), int(offset[1]), 0), order=order)

        # Get the new image
        new_image = sitk.GetImageFromArray(np.transpose(image_array, axes=(2, 1, 0)))

        # Set the new spacing, direction and origin
        new_image.SetSpacing(spacing)
        new_image.SetDirection(direction)
        new_image.SetOrigin(origin)

        # Return the new image
        return new_image
    
    # Adjust the image
    def adjust_image(self, image, gamma=np.random.uniform(1, 2)):

        # Get the image array
        image_array = np.transpose(sitk.GetArrayFromImage(image), axes=(2, 1, 0))

        # Get the spacing, direction and origin of the image
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        # Adjust the image
        image_array = (((image_array - image_array.min()) / (image_array.max() - image_array.min())) ** gamma) * (255 - 0) + 0

        # Get the new image
        new_image = sitk.GetImageFromArray(np.transpose(image_array, axes=(2, 1, 0)))

        # Set the new spacing, direction and origin
        new_image.SetSpacing(spacing)
        new_image.SetDirection(direction)
        new_image.SetOrigin(origin)

        # Return the new image
        return new_image
    

# Pad the image and tractogram
class Pad(object):
    """
    Add padding to the image and tractogram if size smaller than patch size

    Args:
        patch_size (int or tuple) - The patch size to pad to
        -> Int - isotropic padding
        -> Tuple - anisotropic padding
    """

    # 