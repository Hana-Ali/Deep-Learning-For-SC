import os
import torch
import torch.nn as nn
import numpy as np
from utils import *

from pytorch_lightning import LightningModule

# Defining the lightning module 
class LitResNet(LightningModule):
    
    # Constructor
    def __init__(self, encoder, criterion, epoch, losses_path, residual_arrays_path, 
                 separate_hemisphere, n_gpus, use_amp):
        
        # Initialize the parent class
        super(LitResNet, self).__init__()
        
        # Initialize the self attributes
        self.model = encoder
        self.criterion = criterion
        self.epoch = epoch
        self.residual_arrays_path = residual_arrays_path
        self.losses_path = losses_path
        self.separate_hemisphere = separate_hemisphere
        self.n_gpus = n_gpus
        self.use_amp = use_amp
        
        # Activate manual optimization
        self.automatic_optimization = False
        
    # Define the optimizers
    def configure_optimizers(self):
        
        # Make an adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        # Return the optimizer
        return optimizer
        
    # Define a training step
    def training_step(self, batch, batch_idx):
        
        # Get the data from the batch
        (b0, (residual, is_flipped), injection_center) = batch
        
        # Define indices for the coordinates
        x_coord = 2
        y_coord = 3
        z_coord = 4
        coordinates = [x_coord, y_coord, z_coord]
                
        # Get the midpoint of the x dimension
        x_midpoint = int(residual.shape[x_coord] / 2)

        # If we want to separate by hemisphere
        if self.separate_hemisphere:

            # Define the half shape
            half_shape = (b0.shape[0], b0.shape[1], x_midpoint, b0.shape[3], b0.shape[4])

            # Define the hemisphere tensors with the correct shape
            b0_hemisphere = torch.empty(half_shape)
            residual_hemisphere = torch.empty(half_shape)

            # Get the left or right hemisphere, depending on whether it's flipped or not
            for item in range(b0.shape[0]):
                if is_flipped[item]: # Flipped means we go from 256 -> 128 because it's on the left (can check mrtrix to verify this)
                    b0_hemisphere[item, :, :, :, :] = b0[item, :, x_midpoint:, :, :]
                    residual_hemisphere[item, :, :, :, :] = residual[item, :, x_midpoint:, :, :]
                else: # Not flipped means we go from 0 -> 128 because it's on the right (can check mrtrix to verify this)
                    b0_hemisphere[item, :, :, :, :] = b0[item, :, :x_midpoint, :, :]
                    residual_hemisphere[item, :, :, :, :] = residual[item, :, :x_midpoint, :, :]
                    
        # If we don't want to separate by hemisphere, we instead just use the things as they are
        else:
            
            # Define the hemispheres as the inputs themselves
            b0_hemisphere, residual_hemisphere = b0, residual

        # Define the kernel size (cube will be 2 * kernel_size) - HYPERPARAMETER
        kernel_size = 8
        half_kernel = kernel_size // 2

        # Pad the b0 and residuals to be of a shape that is a multiple of the kernel_size
        b0_hemisphere = pad_to_shape(b0_hemisphere, kernel_size)
        residual_hemisphere = pad_to_shape(residual_hemisphere, kernel_size)

        # Get the batch size
        batch_size = b0_hemisphere.shape[0]
                
        # Create a new tensor of size batch_size x 3 x kernel_size x kernel_size x kernel_size, that has the injection centers tiled
        injection_center_tiled = np.tile(injection_center, (kernel_size, kernel_size, kernel_size, 1, 1))
        injection_center_tiled = torch.from_numpy(injection_center_tiled).float()
        injection_center_tiled = torch.permute(injection_center_tiled, (3, 4, 0, 1, 2))

        # Create a tensor of the same shape as the residual hemisphere
        predictions_array = np.zeros_like(residual_hemisphere.numpy())

        # Get the start and end indices, as well as skipping step size
        overlapping = False
        x_centers, y_centers, z_centers = get_centers(residual_hemisphere, kernel_size, overlapping)
                
        # Initialize and get an optimizer
        optimizer = self.optimizers()
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Create losses list
        losses = []

        # For every x_center
        for x in x_centers:

            # For every y_center
            for y in y_centers:

                # For every z_center
                for z in z_centers:

                    # Get the x, y, z coordinate into a list
                    curr_coord = [x, y, z]
                    
                    # Tile the coordinates into the appropriate shape
                    image_coordinates = np.tile(np.array(curr_coord), (kernel_size, kernel_size, kernel_size, batch_size, 1))
                    image_coordinates = torch.from_numpy(image_coordinates).float()
                    image_coordinates = torch.permute(image_coordinates, (3, 4, 0, 1, 2))
                    
                    # Get the cube in the residual that corresponds to this coordinate
                    residual_cube = grab_cube_around_voxel(image=residual_hemisphere, voxel_coordinates=[x, y, z], kernel_size=int(kernel_size / 2))
                    
                    # Get the cube in the DWI that corresponds to this coordinate
                    b0_cube = grab_cube_around_voxel(image=b0_hemisphere, voxel_coordinates=[x, y, z], kernel_size=kernel_size)
                    
                    # Turn the cubes into tensors
                    residual_cube = torch.from_numpy(residual_cube).float()
                    b0_cube = torch.from_numpy(b0_cube).float()
                                        
                    # Get the model output
                    (predicted_residual, loss, batch_size)  = batch_loss(self.model, b0_cube, injection_center_tiled, image_coordinates, 
                                                                         residual_cube, self.criterion,
                                                                         n_gpus=self.n_gpus, use_amp=self.use_amp)
                    
                    # Get the residual as a numpy array
                    predicted_residual = predicted_residual.cpu().detach().numpy()

                    # Get the start of indexing for this new array
                    (start_idx_x, start_idx_y, start_idx_z,
                     end_idx_x, end_idx_y, end_idx_z) = get_predictions_indexing(x, y, z, half_kernel, predictions_array)
                    
                    # Store the predicted residual in the correct array of the predictions array
                    predictions_array[:, :, start_idx_x : end_idx_x,
                                            start_idx_y : end_idx_y,
                                            start_idx_z : end_idx_z] = predicted_residual

                    # Delete the residual just for space
                    del predicted_residual

                    # Empty cache
                    if self.n_gpus:
                        torch.cuda.empty_cache()

                    # Update the loss
                    losses.append(loss.item())

                    # Compute the gradients
                    self.manual_backward(loss)
                    
                    # Clip the gradient
                    self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

                    # Update the parameters
                    optimizer.step()

                    # Delete the loss
                    del loss
                    
                    # Dictionary
                    self.log_dict({"loss_curr": loss, "loss_avg": np.average(np.array(losses))}, prog_bar=True)
                    
        # Dump the predicted residuals array
        print("Saving...")
        predictions_folder = os.path.join(self.residual_arrays_path, "train", "epoch_{}".format(self.epoch))
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)
        prediction_filename = os.path.join(predictions_folder, "batch_{}.npy".format(batch_idx))
        np.save(prediction_filename, predictions_array)

        print("Saving average losses...")
        losses_folder = os.path.join(self.losses_path, "train", "epoch_{}".format(self.epoch))
        if not os.path.exists(losses_folder):
            os.makedirs(losses_folder)
        losses_filename = os.path.join(losses_folder, "losses_{}.npy".format(batch_idx))
        np.save(losses_filename, np.average(np.array(losses)))

    # Define the validation loop
    def validation_step(self, batch, batch_idx):

        # Get the data from the batch
        (b0, (residual, is_flipped), injection_center) = batch
        
        # Define indices for the coordinates
        x_coord = 2
        y_coord = 3
        z_coord = 4
        coordinates = [x_coord, y_coord, z_coord]
                
        # Get the midpoint of the x dimension
        x_midpoint = int(residual.shape[x_coord] / 2)

        # If we want to separate by hemisphere
        if self.separate_hemisphere:

            # Define the half shape
            half_shape = (b0.shape[0], b0.shape[1], x_midpoint, b0.shape[3], b0.shape[4])

            # Define the hemisphere tensors with the correct shape
            b0_hemisphere = torch.empty(half_shape)
            residual_hemisphere = torch.empty(half_shape)

            # Get the left or right hemisphere, depending on whether it's flipped or not
            for item in range(b0.shape[0]):
                if is_flipped[item]: # Flipped means we go from 256 -> 128 because it's on the left (can check mrtrix to verify this)
                    b0_hemisphere[item, :, :, :, :] = b0[item, :, x_midpoint:, :, :]
                    residual_hemisphere[item, :, :, :, :] = residual[item, :, x_midpoint:, :, :]
                else: # Not flipped means we go from 0 -> 128 because it's on the right (can check mrtrix to verify this)
                    b0_hemisphere[item, :, :, :, :] = b0[item, :, :x_midpoint, :, :]
                    residual_hemisphere[item, :, :, :, :] = residual[item, :, :x_midpoint, :, :]
                    
        # If we don't want to separate by hemisphere, we instead just use the things as they are
        else:
            
            # Define the hemispheres as the inputs themselves
            b0_hemisphere, residual_hemisphere = b0, residual

        # Define the kernel size (cube will be 2 * kernel_size) - HYPERPARAMETER
        kernel_size = 8
        half_kernel = kernel_size // 2

        # Pad the b0 and residuals to be of a shape that is a multiple of the kernel_size
        b0_hemisphere = pad_to_shape(b0_hemisphere, kernel_size)
        residual_hemisphere = pad_to_shape(residual_hemisphere, kernel_size)

        # Get the batch size
        batch_size = b0_hemisphere.shape[0]
                
        # Create a new tensor of size batch_size x 3 x kernel_size x kernel_size x kernel_size, that has the injection centers tiled
        injection_center_tiled = np.tile(injection_center, (kernel_size, kernel_size, kernel_size, 1, 1))
        injection_center_tiled = torch.from_numpy(injection_center_tiled).float()
        injection_center_tiled = torch.permute(injection_center_tiled, (3, 4, 0, 1, 2))

        # Create a tensor of the same shape as the residual hemisphere
        predictions_array = np.zeros_like(residual_hemisphere.numpy())

        # Get the start and end indices, as well as skipping step size
        overlapping = False
        x_centers, y_centers, z_centers = get_centers(residual_hemisphere, kernel_size, overlapping)
                        
        # Create losses list
        losses = []

        # For every x_center
        for x in x_centers:

            # For every y_center
            for y in y_centers:

                # For every z_center
                for z in z_centers:

                    # Get the x, y, z coordinate into a list
                    curr_coord = [x, y, z]
                    
                    # Tile the coordinates into the appropriate shape
                    image_coordinates = np.tile(np.array(curr_coord), (kernel_size, kernel_size, kernel_size, batch_size, 1))
                    image_coordinates = torch.from_numpy(image_coordinates).float()
                    image_coordinates = torch.permute(image_coordinates, (3, 4, 0, 1, 2))
                    
                    # Get the cube in the residual that corresponds to this coordinate
                    residual_cube = grab_cube_around_voxel(image=residual_hemisphere, voxel_coordinates=[x, y, z], kernel_size=int(kernel_size / 2))
                    
                    # Get the cube in the DWI that corresponds to this coordinate
                    b0_cube = grab_cube_around_voxel(image=b0_hemisphere, voxel_coordinates=[x, y, z], kernel_size=kernel_size)
                    
                    # Turn the cubes into tensors
                    residual_cube = torch.from_numpy(residual_cube).float()
                    b0_cube = torch.from_numpy(b0_cube).float()
                                        
                    # Get the model output
                    (predicted_residual, loss, batch_size)  = batch_loss(self.model, b0_cube, injection_center_tiled, image_coordinates, 
                                                                         residual_cube, self.criterion,
                                                                         n_gpus=self.n_gpus, use_amp=self.use_amp)
                    
                    # Get the residual as a numpy array
                    predicted_residual = predicted_residual.cpu().detach().numpy()

                    # Get the start of indexing for this new array
                    (start_idx_x, start_idx_y, start_idx_z,
                     end_idx_x, end_idx_y, end_idx_z) = get_predictions_indexing(x, y, z, half_kernel, predictions_array)
                    
                    # Store the predicted residual in the correct array of the predictions array
                    predictions_array[:, :, start_idx_x : end_idx_x,
                                            start_idx_y : end_idx_y,
                                            start_idx_z : end_idx_z] = predicted_residual

                    # Delete the residual just for space
                    del predicted_residual

                    # Empty cache
                    if self.n_gpus:
                        torch.cuda.empty_cache()

                    # Update the loss
                    losses.append(loss.item())
                    
                    # Dictionary
                    self.log_dict({"val_loss_curr": loss, "val_loss_avg": np.average(np.array(losses))}, prog_bar=True)
                    
        # Dump the predicted residuals array
        print("Saving...")
        predictions_folder = os.path.join(self.residual_arrays_path, "val")
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)
        prediction_filename = os.path.join(predictions_folder, "batch_{}.npy".format(batch_idx))
        np.save(prediction_filename, predictions_array)

        print("Saving average losses...")
        losses_folder = os.path.join(self.losses_path, "val")
        if not os.path.exists(losses_folder):
            os.makedirs(losses_folder)
        losses_filename = os.path.join(losses_folder, "losses_{}.npy".format(batch_idx))
        np.save(losses_filename, np.average(np.array(losses)))
