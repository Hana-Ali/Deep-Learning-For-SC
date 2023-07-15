import SimpleITK as sitk

# This class will be used to convert the tractogram to a binary image
class BinaryThreshold:

    # This function will be used to initialize the class
    def __init__(self, lower_threshold, upper_threshold, inside_value=1, outside_value=0):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.inside_value = inside_value
        self.outside_value = outside_value

    # This function will be used to convert the tractogram to a binary image
    def __call__(self, tractogram):
        
        # Convert the tractogram to a binary image
        tractogram = sitk.BinaryThreshold(tractogram, self.lower_threshold, self.upper_threshold, self.inside_value, self.outside_value)

        return tractogram

# This class will be used to mask the image with the tractogram
class MaskImage:
    
        # This function will be used to initialize the class
        def __init__(self, tractogram):
            self.tractogram = tractogram
    
        # This function will be used to mask the image with the tractogram
        def __call__(self, image):
    
            # Convert the tractogram to a binary image
            tractogram = BinaryThreshold(1, 255)(self.tractogram)

            # Save the binary tractogram
            sitk.WriteImage(tractogram, "/mnt/c/tractography/combined_normal_test/tracer_streamlines_binary.nii.gz")
    
            # Mask the image with the tractogram
            image_masked = sitk.Mask(image, tractogram)
    
            return image_masked