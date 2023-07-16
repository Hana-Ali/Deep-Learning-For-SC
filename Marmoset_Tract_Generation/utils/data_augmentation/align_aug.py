import SimpleITK as sitk

# Align an image to a reference image
class Align(object):

    # Constructor
    def __init__(self, image, reference):
            
        # Define the name of the class
        self.name = 'Align'

        # Set the image and reference
        self.image = image
        self.reference = reference

    # Call function
    def __call__(self):

        # Get the array from the image
        image_array = sitk.GetArrayFromImage(self.image)

        # Get origin, direction and spacing of reference
        reference_origin = self.reference.GetOrigin()
        reference_direction = self.reference.GetDirection()
        reference_spacing = self.reference.GetSpacing()

        # Convert image array to image and set origin, direction and spacing
        image = sitk.GetImageFromArray(image_array)
        image.SetOrigin(reference_origin)
        image.SetDirection(reference_direction)
        image.SetSpacing(reference_spacing)

        # Return the image
        return image
