"""
Here we define the configurations that are shared between training and testing of the model.

This includes things such as:
    - Training_data path (where the training data is stored)
    - Testing_data path (where the testing data is stored)
    - Number of input channels (number of channels in the DWI image)
    - Number of output channels (number of channels in the tractogram)
"""

# Import the necessary packages
import os
import utils
import torch
import models
import argparse

# Define the shared configurations
class SharedConfigs():

    # Constructor
    def __init__(self):
        # Initialize to false
        self.initialized = False

    # Initialize the shared configurations
    def initialize(self, parser):
        
        # Data paths
        parser.add_argument('--training_data_path', type=str, default='data/train', help='path to training data')
        parser.add_argument('--testing_data_path', type=str, default='data/test', help='path to testing data')

        # Model stuff
        parser.add_argument('--model', type=str, default='cyclegan', help='chooses which model to use')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='selects model to use for netD')
        parser.add_argument('--netD_n_layers', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='selects model to use for netG')

        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        parser.add_argument('--patching_size', default=[128, 128, 64], help='patching size for the input image')

        # Training stuff
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--drop_ratio', default=0, help='Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

        # Data augmentation stuff
        parser.add_argument('--resample', default=False, help='Decide or not to rescale the images to a new resolution')
        parser.add_argument('--new_resolution', default=(0.45, 0.45, 0.45), help='New resolution (if you want to resample the data again during training')
        parser.add_argument('--min_pixel', default=0.1, help='Percentage of minimum non-zero pixels in the cropped label')

        # Logistics
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA (keep it AtoB)')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--gpu_ids', default='2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')

        # Extra stuff
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')

        # Set initialized to true
        self.initialized = True

        # Return the parser
        return parser
    
    # Get the shared configurations
    def get_shared_configs(self):
        # Initialize parser with basic options
        if not self.initialized:
            # Initialize the parser
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            # Initialize the parser with the shared configurations
            parser = self.initialize(parser)

        # Get the basic options
        config, _ = parser.parse_known_args()

        # Modify model-related parser options
        model_name = config.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # Parse the arguments
        config, _ = parser.parse_known_args()

        # Set the parser to the new one
        self.parser = parser

        # Return parsed configs
        return parser.parse_args()
    
    # Print the shared configurations
    def print_shared_configs(self, config):
        # Initialize the message
        message = ''
        message += '------------ Config -------------'
        # For every item in the configuration
        for k, v in sorted(vars(config).items()):
            # Get the default
            comment = ''
            default = self.parser.get_default(k)
            # If the value is not the default
            if v != default:
                # Add the comment
                comment = '\t[default: %s]' % str(default)
            # Add the message
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------- End ----------------'
        
        # Print the final message
        print(message)

        # Create the experiment directory
        experiment_directory = os.path.join(config.checkpoints_dir, config.name)
        utils.check_directory(experiment_directory)

        # Save the config to a file
        file_name = os.path.join(experiment_directory, 'config.txt')
        with open(file_name, 'wt') as config_file:
            config_file.write(message)
            config_file.write('\n')

    # Parse the shared configurations
    def parse_shared_configs(self):

        # Get the shared configurations
        config = self.get_shared_configs()
        # Set to training
        config.isTrain = self.isTrain

        # Process the suffix
        if config.suffix:
            # Get the suffix
            suffix = ('_' + config.suffix.format(**vars(config))) if config.suffix != '' else ''
            # Set the name
            config.name = config.name + suffix

        # Print the options
        self.print_shared_configs(config)

        # Set the GPU ids
        gpu_string_ids = list(config.gpu_ids.remove(','))
        config.gpu_ids = []
        # For each GPU id
        for gpu_id in gpu_string_ids:
            # Get the id
            int_gpu_id = int(gpu_id)
            # If the id is greater than 0
            if int_gpu_id >= 0:
                # Add the id
                config.gpu_ids.append(int_gpu_id)
        # If there are GPU ids
        if len(config.gpu_ids) > 0:
            # Set the GPU ids
            torch.cuda.set_device(config.gpu_ids[0])

        # Set the config
        self.config = config

        # Return the config
        return self.config

