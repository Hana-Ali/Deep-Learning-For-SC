# Worked from 10 -> 1:30
# Idea is to generate nii.gz and then find only the parts that are compatible with the mask
# This is done by multiplying the mask with the generated image
# then this is what we use to train the discriminator


# The way we organize this is: if the data falls into any of the tracer or tracer_sharp bundles,
# then it wil fall within ground truth and that's what we compare it to. If it falls outside,
# then that's within the merged_symm dwi tractography bundle, and that's what we compare it to.

# Define the generator and discriminator
# Define the loss function
# Define the optimizer
# Define the training loop
# Define the testing loop

