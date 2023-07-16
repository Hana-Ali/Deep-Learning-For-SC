import numpy as np
import SimpleITK as sitk

_interpolator_image = 'linear'          # interpolator image
_interpolator_label = 'linear'          # interpolator label

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
            # self.new_voxel_size = (new_voxel_size, new_voxel_size, new_voxel_size)
            self.new_voxel_size = new_voxel_size
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
