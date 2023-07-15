import SimpleITK as sitk

# Pad the image and tractogram
class Padding(object):
    """
    Add padding to the image and tractogram if size smaller than patch size

    Args:
        patch_size (int or tuple) - The patch size to pad to
        -> Int - isotropic padding
        -> Tuple - anisotropic padding
    """

    # Constructor
    def __init__(self, patch_size):

        # Define the name of the class
        self.name = 'Padding'

        # Check what the patch size is
        if isinstance(patch_size, int):
            # Set the patch size
            self.patch_size = (patch_size, patch_size, patch_size)
        else:
            assert len(patch_size) == 3, 'Patch size should be a tuple of length 3'
            # Set the patch size
            self.patch_size = patch_size

        # Assert that all values are > 0
        assert all(i > 0 for i in self.patch_size), 'All values in patch size should be > 0'

    # Call function
    def __call__(self, sample):

        # Get the image and the tractogram
        image, tractogram = sample['image'], sample['tractogram']

        # Get the old size
        old_size = image.GetSize()

        # If the old size is bigger than the patch size
        if old_size[0] >= self.patch_size[0] and old_size[1] >= self.patch_size[1] and old_size[2] >= self.patch_size[2]:
            # Return the image and tractogram
            return sample
        else:
            # Make the patch size a list
            patch_size = list(self.patch_size)

            # Set the new sizes depending on which is bigger
            new_size = [max(old_size[0], patch_size[0]), max(old_size[1], patch_size[1]), max(old_size[2], patch_size[2])]

            # Make into a tuple
            new_size = tuple(new_size)

            # Define the resampler
            resampler = sitk.ResampleImageFilter()

            # Set properties of the resampler
            resampler.SetInterpolator(sitk.sitkBSpline)
            resampler.SetOutputSpacing(image.GetSpacing())
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetSize(new_size)

            # Execute the resampler
            image = resampler.Execute(image)

            # Resample the tractogram
            resampler.SetInterpolator(sitk.sitkBSpline)
            resampler.SetOutputOrigin(tractogram.GetOrigin())
            resampler.SetOutputDirection(tractogram.GetDirection())

            # Execute the resampler
            tractogram = resampler.Execute(tractogram)

            # Return the image and tractogram
            return {'image': image, 'tractogram': tractogram}
