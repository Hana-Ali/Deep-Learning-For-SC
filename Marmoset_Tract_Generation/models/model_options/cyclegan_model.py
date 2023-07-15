"""
This file is dedicated to the cycleGAN model. It uses the cycleGAN model to generate synthetic DWI images
from a given DWI image and a given tractogram, and the other way around
"""

import torch
import itertools
from .base_model import BaseModel

import sys
sys.path.append('..')
from model_builders.network_funcs import *

# Define the cycleGAN model
class cycleGANModel(BaseModel):

    # Constructor
    def __init__(self, config):
        
        # Call the constructor of the base class
        super(cycleGANModel, self).__init__(config)

        # Specify the training losses to print. This calls base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        # Specify the training visualizations to print. This calls base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        # If the model is in training mode
        if self.isTrain and self.config.lambda_identity > 0.0:
            # Add the identity loss
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')
        
        # Add the visualizations to the visual names list
        self.visual_names = visual_names_A + visual_names_B

        # Specify the models to save to disk. This calls base_model.save_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        # Define the networks
        self.netG_A = define_G(config.input_nc, config.output_nc, config.ngf, config.netG, config.norm, 
                               not config.no_dropout, config.init_type, config.init_gain, self.gpu_ids)
        self.netG_B = define_G(config.output_nc, config.input_nc, config.ngf, config.netG, config.norm,
                                 not config.no_dropout, config.init_type, config.init_gain, self.gpu_ids)
        
        # If the model is in training mode
        if self.isTrain:
            # Get whether or not to use sigmoid
            use_sigmoid = config.no_lsgan
            # Define the discriminators
            self.netD_A = define_D(config.output_nc, config.ndf, config.netD,
                                    config.n_layers_D, config.norm, use_sigmoid, config.init_type, config.init_gain, self.gpu_ids)
            self.netD_B = define_D(config.input_nc, config.ndf, config.netD,
                                    config.n_layers_D, config.norm, use_sigmoid, config.init_type, config.init_gain, self.gpu_ids)
            
        # If the model is in training mode
        if self.isTrain:
            # Get the image pool
            self.fake_A_pool = ImagePool(config.pool_size)
            self.fake_B_pool = ImagePool(config.pool_size)

            # Define the loss functions
            self.criterionGAN = GANLoss(use_lsgan=not config.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # Define the optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=config.lr, betas=(config.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=config.lr, betas=(config.beta1, 0.999))
            
            # Add the optimizers to the optimizers list
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    ##############################################################
    ############### Forward and Backward Functions ###############
    ##############################################################

    # Define the forward function
    def forward(self):
        
        # For fake B and reconstruct A
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        # For fake A and reconstruct B
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    # Define the backward function
    def backward_D_basic(self, netD, real, fake):

        # Get the prediction of the real image
        pred_real = netD(real)
        # Calculate the loss of the real image
        loss_D_real = self.criterionGAN(pred_real, True)

        # Get the prediction of the fake image
        pred_fake = netD(fake.detach())
        # Calculate the loss of the fake image
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Calculate the total loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # Calculate the gradients
        loss_D.backward()

        # Return the loss
        return loss_D
    
    # Define the backward function for A
    def backward_D_A(self):

        # Get the fake B image
        fake_B = self.fake_B_pool.query(self.fake_B)
        # Calculate the loss
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    # Define the backward function for B
    def backward_D_B(self):
            
        # Get the fake A image
        fake_A = self.fake_A_pool.query(self.fake_A)
        # Calculate the loss
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    # Define the backward function for G
    def backward_G(self):
        
        # Get the lambda values
        lambda_idt = self.config.lambda_identity
        lambda_A = self.config.lambda_A
        lambda_B = self.config.lambda_B

        # Get the co_A and co_B values
        lambda_co_A = self.config.lambda_co_A
        lambda_co_B = self.config.lambda_co_B

        # If the lambda values are not 0, identity loss is used
        if lambda_idt > 0:
            
            # G_A should be identity if real_B is fed
            self.idt_A = self.netG_A(self.real_B)
            # Calculate the identity loss
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt

            # G_B should be identity if real_A is fed
            self.idt_B = self.netG_B(self.real_A)
            # Calculate the identity loss
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt

        # If the lambda values are not 0, cycle loss is used
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # Define the GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # Define the GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Define the forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        # Define the backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Define the correlation coefficient loss
        self.loss_corr_coeff_GA = correlation_coefficient_loss(self.fake_B, self.real_A) * lambda_co_A # fake ct & real mr; Evaluate the Generator of ct(G_A)
        self.loss_corr_coeff_GB = correlation_coefficient_loss(self.fake_A, self.real_B) * lambda_co_B # fake mr & real ct; Evaluate the Generator of mr(G_B)

        # Calculate the total loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_corr_coeff_GA + self.loss_corr_coeff_GB

        # Calculate the gradients
        self.loss_G.backward()

    # Define the optimize parameters function
    def optimize_parameters(self):
            
        # Forward pass
        self.forward()

        # Update G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # Update D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    ##############################################################
    ####################### Misc Functions #######################
    ##############################################################

    # Define the name
    def name(self):
        return 'cycleGANModel'
    
    # Set the input
    def set_input(self, input):
        # Find which direction the model is going
        AtoB = self.config.direction == 'AtoB'
        
        # Get the input images
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)


