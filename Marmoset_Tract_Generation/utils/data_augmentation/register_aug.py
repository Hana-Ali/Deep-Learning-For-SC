import SimpleITK as sitk

# Define the class
class Register(object):

    # Constructor
    def __init__(self, image, label):
            
        # Define the name of the class
        self.name = 'Register'

        # Set the image and label
        self.image = image
        self.label = label

    # Call function
    def __call__(self):

        # Copy the image and label
        image, image_sobel, label, label_sobel = self.image, self.image, self.label, self.label

        # Define a Gaussian filter
        gaussian_filter = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
        
        # Apply the filter
        image_sobel = gaussian_filter.Execute(image_sobel)
        label_sobel = gaussian_filter.Execute(label_sobel)

        # Define fixed and moving images for registration
        fixed_image = label_sobel
        moving_image = image_sobel

        # Define the initial transform
        initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                        moving_image,
                                                        sitk.Euler3DTransform(),
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)

        # Define the registration method
        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings.
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.1)
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer settings.
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Don't optimize in-place, we would possibly like to run this cell multiple times.
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Execute the registration
        final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                    sitk.Cast(moving_image, sitk.sitkFloat32))

        # Resample the image
        image = sitk.Resample(image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                        moving_image.GetPixelID())

        # Return the image and label
        return image, label

