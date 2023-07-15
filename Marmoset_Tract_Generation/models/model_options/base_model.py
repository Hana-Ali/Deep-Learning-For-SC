"""
This can be understood as an abstract class, which the cycleGAN will inherit from.
"""

import torch
import os
from collections import OrderedDict

import sys
from model_builders.network_funcs import *

# Define the base model
class BaseModel():

    # Constructor
    def __init__(self, config):
        self.config = config
        self.gpu_ids = config.gpu_ids
        self.isTrain = config.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(config.checkpoints_dir, config.name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []

    ###############################################################
    ###################### Getter Functions #######################
    ###############################################################

    # Return the visualization images (?)
    def get_current_visuals(self):
        # Create a dictionary
        visuals = OrderedDict()
        # For each visual name in the visual name list
        for name in self.visual_names:
            # Get the visual name
            if isinstance(name, str):
                visuals[name] = getattr(self, name)
        # Return the visuals
        return visuals
    
    # Return the loss values
    def get_current_losses(self):
        # Create a dictionary
        errors_ret = OrderedDict()
        # For each loss name in the loss name list
        for name in self.loss_names:
            # Get the loss name
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        # Return the errors
        return errors_ret

    # Define the name of the model
    def get_name(self):
        return 'BaseModel'
    
    # Get the image paths
    def get_image_paths(self):
        return self.image_paths
    
    ###############################################################
    ################# Setup and Saving Functions ##################
    ###############################################################

    # Setup the network depending on whether we are training or testing
    def setup(self, config):
        
        # If we are training, we need to set up the training mode
        if self.isTrain:
            # Define the schedulers
            self.schedulers = [get_scheduler(optimizer, config) for optimizer in self.optimizers]
        
        # If we are testing, we need to set up the testing mode
        if not self.isTrain or config.continue_train:
            # Load the network
            self.load_networks(config.which_epoch)

        # Print the network, if verbose
        self.print_networks(config.verbose)
        
    # Save the network
    def save_networks(self, epoch):
        # For each model name in the model name list
        for name in self.model_names:
            # Get the model name
            if isinstance(name, str):
                # Get the network
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                # Get the network
                net = getattr(self, 'net' + name)
                # If there are multiple GPUs, get the first GPU
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                # If there is only one GPU, get the first GPU
                else:
                    torch.save(net.cpu().state_dict(), save_path)
                # # Get the network
                # net.to(self.device)

    # Load the network
    def load_networks(self, epoch):
        # For each model name in the model name list
        for name in self.model_names:
            # Get the model name
            if isinstance(name, str):
                # Get the network
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                # Get the network
                net = getattr(self, 'net' + name)
                # If there are multiple GPUs, get the first GPU
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                # Print the network
                print('loading the model from %s' % load_path)
                # Load the network
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                net.load_state_dict(state_dict)

    # Function to patch the instance norm state dictionary
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        # Get the key
        key = keys[i]
        # If we are at the end of the keys, pointing to parameter or buffer
        if i + 1 == len(keys):
            # If the class is instance norm, and key is running mean or running var
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                # If the key is not in the state dictionary
                if getattr(module, key) is None:
                    # Set the key to be the state dictionary
                    state_dict.pop('.'.join(keys))
            # If the class is instance norm, but the key is num batches tracked
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                # Set the key to be the state dictionary
                state_dict.pop('.'.join(keys))
        # If we are not at the end of the keys, pointing to submodule
        else:
            # Apply this function
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    ###############################################################
    ############## Testing and Evaluation Functions ###############
    ###############################################################

    # Set model to no grad mode
    def set_testing(self):
        with torch.no_grad():
            self.forward()

    # Set model to evaluation mode
    def set_eval(self):       
        # For each model in the model list
        for name in self.model_names:
            if isinstance(name, str):
                # Get the model name
                net = getattr(self, 'net' + name)
                net.eval()
    
    # Function to set requires grad
    def set_requires_grad(self, nets, requires_grad=False):
        # If nets is not a list, make it a list
        if not isinstance(nets, list):
            nets = [nets]
        # For each net in the nets list
        for net in nets:
            # If the net is not None
            if net is not None:
                # Set the requires grad
                for param in net.parameters():
                    param.requires_grad = requires_grad

##############################################################
####################### Misc Functions #######################
##############################################################

    # Update the learning rate
    def update_learning_rate(self):
        # For each scheduler in the scheduler list
        for scheduler in self.schedulers:
            # Update the learning rate
            scheduler.step()
        # Get the learning rate and print it
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # Print the network
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        # For each model name in the model name list
        for name in self.model_names:
            # Get the model name
            if isinstance(name, str):
                # Get the network
                net = getattr(self, 'net' + name)
                # Print the network
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # Modify the parser to add command line configuration for the model
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
            
