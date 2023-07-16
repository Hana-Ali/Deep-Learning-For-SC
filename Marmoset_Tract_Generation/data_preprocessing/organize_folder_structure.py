import os
import regex as re
import argparse
import numpy as np
import SimpleITK as sitk
from utils import *

# Define the parser
parser = argparse.ArgumentParser(description='Organize the folder structure of the data')

# Define the arguments - remember labels is what we want to generate
parser.add_argument('--images', default='/mnt/d/GAN_Data/DWI', help='path to the images a (early frames)')
parser.add_argument('--labels', default='/mnt/d/GAN_Data/Tractograms', help='path to the images b (late frames)')
parser.add_argument('--split', default=50, help='number of images for testing')
parser.add_argument('--resolution', default=(1.6,1.6,1.6), help='new resolution to resample the all data')
args = parser.parse_args()

# Main 
if __name__ == '__main__':

    # List the images and labels
    list_images = list_files(args.images)
    list_labels = list_files(args.labels)

    # Setting reference image to have all data in the same reference
    reference_image = list_labels[0]
    reference_image = sitk.ReadImage(reference_image)
    # Define a resampler and resample the image
    resampler = Resample(new_voxel_size=args.resolution)
    reference_image = resampler.resample_sitk_image(reference_image, spacing=args.resolution, interpolator='linear')

    # Create the training and testing folders
    main_data_path = "/mnt/d/GAN_Data"
    train_data_path = os.path.join(main_data_path, "train")
    test_data_path = os.path.join(main_data_path, "test")
    check_directory(train_data_path)
    check_directory(test_data_path)

    # Create the folder for the image and label - TRAINING
    save_directory_images_train = os.path.join(train_data_path, "images")
    save_directory_labels_train = os.path.join(train_data_path, "labels")
    check_directory(save_directory_images_train)
    check_directory(save_directory_labels_train)

    # Create the folder for the image and label - TESTING
    save_directory_images_test = os.path.join(test_data_path, "images")
    save_directory_labels_test = os.path.join(test_data_path, "labels")
    check_directory(save_directory_images_test)
    check_directory(save_directory_labels_test)


    # Separate the images into training
    for i in range(len(list_images) - int(args.split)):

        # Get the image and label
        train_image = list_images[int(args.split) + i]
        train_label = list_labels[int(args.split) + i]

        # Print the image and label
        print("Training image: {}".format(train_image))
        print("Training label: {}".format(train_label))

        # Read the image and label
        train_image = sitk.ReadImage(train_image)
        train_label = sitk.ReadImage(train_label)

        # Register the image and label
        label, reference_image = Register(label, reference_image)()
        image, label = Register(image, label)()

        # Resample the image and label
        image = resampler.resample_sitk_image(image, spacing=args.resolution, interpolator='linear')
        label = resampler.resample_sitk_image(label, spacing=args.resolution, interpolator='linear')

        # Create the name of the image and label
        image_directory = os.path.join(save_directory_images_train, "{}.nii.gz".format(i))
        label_directory = os.path.join(save_directory_labels_train, "{}.nii.gz".format(i))

        print("Image directory: {}".format(image_directory))
        print("Label directory: {}".format(label_directory))

        # Write the image and label
        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)

    # Separate the images into testing
    for i in range(int(args.split)):

        # Get the image and label
        test_image = list_images[i]
        test_label = list_labels[i]

        # Print the image and label
        print("Testing image: {}".format(test_image))
        print("Testing label: {}".format(test_label))

        # Read the image and label
        test_image = sitk.ReadImage(test_image)
        test_label = sitk.ReadImage(test_label)

        # Register the image and label
        label, reference_image = Register(label, reference_image)()
        image, label = Register(image, label)()

        # Resample the image and label
        image = resampler.resample_sitk_image(image, spacing=args.resolution, interpolator='linear')
        label = resampler.resample_sitk_image(label, spacing=args.resolution, interpolator='linear')

        # Create the name of the image and label
        image_directory = os.path.join(save_directory_images_test, "{}.nii.gz".format(i))
        label_directory = os.path.join(save_directory_labels_test, "{}.nii.gz".format(i))

        print("Image directory: {}".format(image_directory))
        print("Label directory: {}".format(label_directory))

        # Write the image and label
        sitk.WriteImage(image, image_directory)
        sitk.WriteImage(label, label_directory)




