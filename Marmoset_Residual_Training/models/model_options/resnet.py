import torch
from models.model_builders import *

##############################################################
########################### ResNet ###########################
##############################################################
class ResNet(nn.Module):

    # Constructor
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, 
                 replace_stride_with_dilation=None, norm_layer=None, n_features=3):
        super(ResNet, self).__init__()

        # Set norm layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        # Set the number of input features
        self.in_planes = 64

        # Set the dilation
        self.dilation = 1

        # If the stride is replaced with dilation, set the dilation
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        # If the length of the replace_stride_with_dilation is not equal to 3, raise an error
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        # Set the number of groups
        self.groups = groups

        # Set the width per group
        self.base_width = width_per_group

        # Set the conv1
        self.conv1 = nn.Conv3d(n_features, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)

        # Set the norm1
        self.bn1 = norm_layer(self.in_planes)

        # Set the relu
        self.relu = nn.ReLU(inplace=True)

        # Set the maxpool
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Set the layer1
        self.layer1 = self._make_layer(block, 64, layers[0])

        # Set the layer2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])

        # Set the layer3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

        # Set the layer4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # Set the avgpool
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # Set the fc
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize the weights
        for m in self.modules():

            # If the module is conv, initialize the weights
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            # If the module is batch norm, initialize the weights
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:

            # For each module
            for m in self.modules():

                # If the module is basic block
                if isinstance(m, BasicResidualBlock):
                        
                        # Initialize the weight
                        nn.init.constant_(m.bn2.weight, 0)
                
                # If the module is bottleneck block
                elif isinstance(m, Bottleneck):

                        # Initialize the weight
                        nn.init.constant_(m.bn3.weight, 0)

    # Make layer
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
         
        # Set the norm layer
        norm_layer = self._norm_layer

        # Set the downsample
        downsample = None

        # Set the previous dilation
        previous_dilation = self.dilation

        # If the dilate is true
        if dilate:
                 
                # Set the dilation
                self.dilation *= stride
    
                # Set the stride
                stride = 1
            
        # If the stride is not equal to 1 or the number of input planes is not equal to the number of output planes
        if stride != 1 or self.in_planes != planes * block.expansion:
                 
                # Set the downsample
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        # Set the layers
        layers = []

        # Append the block
        layers.append(block(self.in_planes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))

        # Set the number of input planes
        self.in_planes = planes * block.expansion

        # For each block
        for _ in range(1, blocks):
                 
                # Append the block
                layers.append(block(self.in_planes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        # Return the layers
        return nn.Sequential(*layers)
    
    # Forward
    def forward(self, x):
         
        # Set the x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Set the x
        x = self.layer1(x)

        # Set the x
        x = self.layer2(x)

        # Set the x
        x = self.layer3(x)

        # Set the x
        x = self.layer4(x)

        # Set the x
        x = self.avgpool(x)

        # Set the x
        x = torch.flatten(x, 1)

        # Set the x
        x = self.fc(x)

        # Return the x
        return x

# Define all the resnet variants
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet_18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_18', BasicResidualBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet_34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_34', BasicResidualBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet_50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet_101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet_152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext_50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext_50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext_101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext_101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

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
                 n_gpus, use_amp):
        
        # Initialize the parent class
        super(LitResNet, self).__init__()
        
        # Initialize the self attributes
        self.model = encoder
        self.criterion = criterion
        self.epoch = epoch
        self.residual_arrays_path = residual_arrays_path
        self.losses_path = losses_path
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

        # Get the left or right hemisphere, depending on whether it's flipped or not
        if is_flipped: # Flipped means we go from 256 -> 128 because it's on the left (can check mrtrix to verify this)
            b0_hemisphere = b0[:, :, x_midpoint:, :, :]
            residual_hemisphere = residual[:, :, x_midpoint:, :, :]
        else: # Not flipped means we go from 0 -> 128 because it's on the right (can check mrtrix to verify this)
            b0_hemisphere = b0[:, :, :x_midpoint, :, :]
            residual_hemisphere = residual[:, :, :x_midpoint, :, :]

        # Define the kernel size (cube will be 2 * kernel_size) - HYPERPARAMETER
        kernel_size = 8
        half_kernel = kernel_size // 2

        # Pad the b0 and residuals to be of a shape that is a multiple of the kernel_size
        b0_hemisphere = pad_to_shape(b0_hemisphere, kernel_size)
        residual_hemisphere = pad_to_shape(residual_hemisphere, kernel_size)

        # Create a new tensor of size kernel_size x kernel_size x 3, that has the injection center
        injection_center = np.tile(injection_center, (kernel_size, kernel_size, kernel_size, 1))

        # Turn the data into a torch tensor
        injection_center = torch.from_numpy(injection_center).unsqueeze(0).float()
        injection_center = torch.permute(injection_center, (0, 4, 1, 2, 3))

        # Create a tensor of the same shape as the residual hemisphere
        predictions_array = np.zeros_like(residual_hemisphere.numpy().squeeze(0).squeeze(0))
        # print("predictions_array shape: {}".format(predictions_array.shape))

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

                    # Get the x, y, z coordinate into a torch tensor
                    image_coordinates = np.tile(np.array([x, y, z]), (kernel_size, kernel_size, kernel_size, 1))
                    image_coordinates = torch.from_numpy(image_coordinates).unsqueeze(0).float()
                    image_coordinates = torch.permute(image_coordinates, (0, 4, 1, 2, 3))

                    # Get the cube in the residual that corresponds to this coordinate
                    residual_cube = grab_cube_around_voxel(image=residual_hemisphere, voxel_coordinates=[x, y, z], kernel_size=int(kernel_size / 2))

                    # Get the cube in the DWI that corresponds to this coordinate
                    b0_cube = grab_cube_around_voxel(image=b0_hemisphere, voxel_coordinates=[x, y, z], kernel_size=kernel_size)

                    # Turn the cubes into tensors
                    residual_cube = torch.from_numpy(residual_cube).unsqueeze(0).unsqueeze(0).float()
                    b0_cube = torch.from_numpy(b0_cube).unsqueeze(0).unsqueeze(0).float()

                    # Get the model output
                    (predicted_residual, loss, batch_size)  = batch_loss(self.model, b0_cube, injection_center, image_coordinates, 
                                                                             residual_cube, self.criterion,
                                                                             n_gpus=self.n_gpus, use_amp=self.use_amp)
                    
                    # Get the residual as a numpy array
                    predicted_residual = predicted_residual.cpu().detach().numpy().squeeze(0).squeeze(0)

                    # Get the start of indexing for this new array
                    (start_idx_x, start_idx_y, start_idx_z,
                     end_idx_x, end_idx_y, end_idx_z) = get_predictions_indexing(x, y, z, half_kernel, predictions_array)
                    
                    # Store the predicted residual in the correct array of the predictions array
                    predictions_array[start_idx_x : end_idx_x,
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
        prediction_filename = os.path.join(predictions_folder, "image_{}.npy".format(batch_idx))
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

        # Get the left or right hemisphere, depending on whether it's flipped or not
        if is_flipped: # Flipped means we go from 256 -> 128 because it's on the left (can check mrtrix to verify this)
            b0_hemisphere = b0[:, :, x_midpoint:, :, :]
            residual_hemisphere = residual[:, :, x_midpoint:, :, :]
        else: # Not flipped means we go from 0 -> 128 because it's on the right (can check mrtrix to verify this)
            b0_hemisphere = b0[:, :, :x_midpoint, :, :]
            residual_hemisphere = residual[:, :, :x_midpoint, :, :]

        # Define the kernel size (cube will be 2 * kernel_size) - HYPERPARAMETER
        kernel_size = 8
        half_kernel = kernel_size // 2

        # Pad the b0 and residuals to be of a shape that is a multiple of the kernel_size
        b0_hemisphere = pad_to_shape(b0_hemisphere, kernel_size)
        residual_hemisphere = pad_to_shape(residual_hemisphere, kernel_size)

        # Create a new tensor of size kernel_size x kernel_size x 3, that has the injection center
        injection_center = np.tile(injection_center, (kernel_size, kernel_size, kernel_size, 1))

        # Turn the data into a torch tensor
        injection_center = torch.from_numpy(injection_center).unsqueeze(0).float()
        injection_center = torch.permute(injection_center, (0, 4, 1, 2, 3))

        # Create a tensor of the same shape as the residual hemisphere
        predictions_array = np.zeros_like(residual_hemisphere.numpy().squeeze(0).squeeze(0))
        # print("predictions_array shape: {}".format(predictions_array.shape))

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

                    # Get the x, y, z coordinate into a torch tensor
                    image_coordinates = np.tile(np.array([x, y, z]), (kernel_size, kernel_size, kernel_size, 1))
                    image_coordinates = torch.from_numpy(image_coordinates).unsqueeze(0).float()
                    image_coordinates = torch.permute(image_coordinates, (0, 4, 1, 2, 3))

                    # Get the cube in the residual that corresponds to this coordinate
                    residual_cube = grab_cube_around_voxel(image=residual_hemisphere, voxel_coordinates=[x, y, z], kernel_size=int(kernel_size / 2))

                    # Get the cube in the DWI that corresponds to this coordinate
                    b0_cube = grab_cube_around_voxel(image=b0_hemisphere, voxel_coordinates=[x, y, z], kernel_size=kernel_size)

                    # Turn the cubes into tensors
                    residual_cube = torch.from_numpy(residual_cube).unsqueeze(0).unsqueeze(0).float()
                    b0_cube = torch.from_numpy(b0_cube).unsqueeze(0).unsqueeze(0).float()

                    # Get the model output
                    (predicted_residual, loss, batch_size)  = batch_loss(self.model, b0_cube, injection_center, image_coordinates, 
                                                                             residual_cube, self.criterion,
                                                                             n_gpus=self.n_gpus, use_amp=self.use_amp)
                    
                    # Get the residual as a numpy array
                    predicted_residual = predicted_residual.cpu().detach().numpy().squeeze(0).squeeze(0)

                    # Get the start of indexing for this new array
                    (start_idx_x, start_idx_y, start_idx_z,
                     end_idx_x, end_idx_y, end_idx_z) = get_predictions_indexing(x, y, z, half_kernel, predictions_array)
                    
                    # Store the predicted residual in the correct array of the predictions array
                    predictions_array[start_idx_x : end_idx_x,
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
        prediction_filename = os.path.join(predictions_folder, "image_{}.npy".format(batch_idx))
        np.save(prediction_filename, predictions_array)

        print("Saving average losses...")
        losses_folder = os.path.join(self.losses_path, "val")
        if not os.path.exists(losses_folder):
            os.makedirs(losses_folder)
        losses_filename = os.path.join(losses_folder, "losses_{}.npy".format(batch_idx))
        np.save(losses_filename, np.average(np.array(losses)))
