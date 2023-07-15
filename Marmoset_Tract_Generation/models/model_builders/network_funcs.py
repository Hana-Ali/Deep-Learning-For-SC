"""
Any additional functions used in the network model or training process
"""

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.nn import init
import functools
from numpy import random

import sys
from model_options.generators import *
from model_options.discriminators import *

# Get the normalization layer
def get_norm_layer(norm_type='instance'):

    # Allowed normalization types
    allowed_norm_types = ['batch', 'instance', 'none']

    # Instance normalization
    if norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=True)
    # Batch normalization
    elif norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    # No normalization
    elif norm_type == 'none':
        norm_layer = None
    # Error
    else:
        raise NotImplementedError('Normalization layer [{type}] is not found. Options are {options}'.format(type=norm_type,
                                                    options=(", ").join(allowed_norm_types)))
    
    # Return the normalization layer
    return norm_layer

# Get the scheduler
def get_scheduler(optimizer, opt):
    
    # Allowed schedulers
    allowed_schedulers = ['lambda', 'step', 'plateau', 'cosine']

    # Lambda scheduler
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    # Step scheduler
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    # Plateau scheduler
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    # Cosine scheduler
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    # Error
    else:
        raise NotImplementedError('Learning rate policy [{policy}] is not implemented. Options are {options}'.format(policy=opt.lr_policy,
                                                                    options=(", ").join(allowed_schedulers)))
    
    # Return the scheduler
    return scheduler

# Initialize the weights
def init_weights(net, init_type='normal', init_gain=0.02):

    # Define the allowed types
    allowed_weight_types = ['normal', 'xavier', 'kaiming', 'orthogonal']

    # Function to initialize the weights
    def init_func(m):
        # Get the name of the class calling this
        classname = m.__class__.__name__
        # Check the class name
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # Check the initialization type
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('Initialization method [{type}] is not implemented. Options are {options}'.format(type=init_type,
                                                                                        options=(", ").join(allowed_weight_types)))
            # Check the bias
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        # Check the normalization layer
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # Apply the initialization function
    print("Initializing network with [{type}]".format(type=init_type))
    net.apply(init_func)

# Initialize the network
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    
    # Get the number of GPUs
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    # Initialize the weights
    init_weights(net, init_type, init_gain=init_gain)
    
    # Return the network
    return net

# Define the generator network
def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    
    # Initialize the network
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    # Allowed generators
    allowed_generators = ['resnet_9blocks', 'resnet_6blocks', 'unet_custom', 'unet_128', 'unet_256']

    # ResNet 9 blocks
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    # ResNet 6 blocks
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    # Custom Unet
    elif netG == 'unet_custom':
        net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # Unet 128
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # Unet 256
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    # Error
    else:
        raise NotImplementedError('Generator model [{model}] not implemented. Options are {types}'.format(model=netG,
                                                                                                          types=(", ").join(allowed_generators)))
    
    # Initialize the network
    return init_net(net, init_type, init_gain, gpu_ids)

# Define the discriminator network
def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):

    # Initialize the network
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    # Allowed discriminators
    allowed_discriminators = ['basic', 'n_layers', 'pixel']

    # Basic discriminator
    if netD == 'basic':
        net = PatchGANDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    # N layers discriminator
    elif netD == 'n_layers':
        net = PatchGANDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    # Pixel discriminator
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    # Error
    else:
        raise NotImplementedError('Discriminator model [{model}] not implemented. Options are {types}'.format(model=netD,
                                                                                                              types=(", ").join(allowed_discriminators)))
    
    # Initialize the network
    return init_net(net, init_type, init_gain, gpu_ids)

# Define the GAN loss
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):

    # Constructor
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
            
        # Initialize the parent class
        super(GANLoss, self).__init__()

        # Allowed GAN modes
        allowed_gan_modes = ['lsgan', 'vanilla']

        # Initialize the attributes
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        # Check the GAN mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        # Check the GAN mode
        elif gan_mode == 'vanilla':
            self.loss = nn.BCELoss()
        # Error
        else:
            raise NotImplementedError('GAN mode [{mode}] not implemented. Options are {types}'.format(mode=gan_mode,
                                                                                                      types=(", ").join(allowed_gan_modes)))    
        
    # Function to get the target tensor
    def get_target_tensor(self, prediction, target_is_real):
            
        # Get the target label
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
                
        # Return the target
        return target_tensor.expand_as(prediction)
    
    # Forward function
    def __call__(self, prediction, target_is_real):
                
        # Get the target tensor
        target_tensor = self.get_target_tensor(prediction, target_is_real)
            
        # Calculate the loss
        loss = self.loss(prediction, target_tensor)
            
        # Return the loss
        return loss
    
# Define the cycle consistency loss - Copilot created
class CycleConsistencyLoss(nn.Module):

    # Constructor
    def __init__(self, lambda_A=10.0, lambda_B=10.0):
            
        # Initialize the parent class
        super(CycleConsistencyLoss, self).__init__()

        # Initialize the attributes
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.loss = nn.L1Loss()
    
    # Forward function
    def __call__(self, real_image, generated_image):
            
        # Calculate the loss
        loss_A = self.loss(generated_image, real_image) * self.lambda_A
        loss_B = self.loss(real_image, generated_image) * self.lambda_B

        # Return the loss
        return loss_A + loss_B
    
# Define the correlation coefficient loss
def correlation_coefficient_loss(prediction, target):
        
    # Get the prediction and target
    x = prediction - torch.mean(prediction)
    y = target - torch.mean(target)
    
    # Calculate the loss
    loss = torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))
    
    # Return the loss
    return 1 - loss**2 # absolute constraint

# Define the image pool
# This class implements an image buffer that stores previously generated images
# This buffer enables us to update discriminators using a history of generated images
# rather than the ones produced by the latest generators
class ImagePool():

    # Constructor
    def __init__(self, pool_size):
                
        # Initialize the parent class
        super(ImagePool, self).__init__()

        # Initialize the attributes
        self.pool_size = pool_size
        
        # If the pool size is greater than 0
        if self.pool_size > 0:
            # Initialize the pool
            self.num_imgs = 0
            self.images = []
        
    # Function to query the pool
    def query(self, images):

        # If the pool size is 0
        if self.pool_size == 0:
            # Return the images
            return images

        # This function selects a random image from the pool, stored here        
        return_images = []
        # For each image
        for image in images:
            
            # Unsqueeze the image
            image = torch.unsqueeze(image.data, 0)
            
            # If the pool is not full
            if self.num_imgs < self.pool_size:
                # Add the image to the pool
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            
            # If the pool is full
            else:
                # Randomly select an image
                prob = random.uniform(0, 1)

                # If the probability is greater than 0.5
                if prob > 0.5:
                    # Select a random image
                    random_id = random.randint(0, self.pool_size - 1)
                    temp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(temp)
                # If the probability is less than 0.5
                else:
                    # Return the image
                    return_images.append(image)
            
            # Concatenate the images
            return_images = torch.cat(return_images, 0)

            # Return the images
            return return_images
