import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model_builders.twoinput_mlp import TwoInputMLP

class AttentionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(AttentionModule, self).__init__()

        # This is inspired by the other one I had before - Squeeze and Excitation
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_x):

        # Apply the squeeze operation
        x = self.squeeze(input_x)

        # Flatten the output of the squeeze operation
        x = x.view(x.size(0), -1)

        # Apply the excitation operation
        x = self.excitation(x)

        # Reshape the output of the excitation operation
        x = x.view(x.size(0), x.size(1), 1, 1, 1)

        # Multiply the input with the output of the excitation operation
        x = input_x * x

        return x

class AttnCNN(nn.Module):
    def __init__(self, channels, filters=[64, 128, 256], reduction=16, output_size=256, n_blocks=1,
                 hidden_size=128, task='classification', contrastive=False, previous=True):

        # Constructor
        super(AttnCNN, self).__init__()

        # Define the number of blocks
        self.n_blocks = n_blocks

        # Define the output size
        self.output_size = output_size

        # Define the number of neurons
        self.neurons = hidden_size

        # Define whether we do previous or not
        self.previous = previous

        # Define whether it's contrastive or not
        self.contrastive = contrastive

        # Define the task
        self.task = task

        # This will store the depthwise separable convolution layers
        depthwise_conv = []

        # Append the first depthwise separable convolution layer
        depthwise_conv += [nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                            nn.BatchNorm3d(channels),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(channels, filters[0], kernel_size=1),
                            nn.BatchNorm3d(filters[0]),
                            nn.ReLU(inplace=True)]
        
        # For the rest of the layers, we use the same structure
        for i in range(len(filters) - 1):

            # Define the depthwise separable convolution layers
            depthwise_conv += [nn.Conv3d(filters[i], filters[i], kernel_size=3, padding=1),
                                nn.BatchNorm3d(filters[i]),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(filters[i], filters[i+1], kernel_size=1),
                                nn.BatchNorm3d(filters[i+1]),
                                nn.ReLU(inplace=True)]
            
        # Make sure the last depthwise separable convolution layer has the same number of channels as the input
        depthwise_conv += [nn.Conv3d(filters[-1], channels, kernel_size=3, padding=1),
                            nn.BatchNorm3d(channels),
                            nn.ReLU(inplace=True)]

        # Convert the list to a sequential model
        self.depthwise_conv = nn.Sequential(*depthwise_conv)

        # Define the attention module
        self.attention_module = AttentionModule(channels, reduction=reduction)

        # Define the global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)

        # Define the final fully connected layers
        self.final_linear = nn.Linear(channels, output_size)

        ########################### PREVIOUS PREDICTIONS ###########################

        # Define the input size of the previous predictions MLP - will always be output * 2
        self.previous_predictions_size = self.output_size * 2
                
        # The flattened size depends on the task
        if self.task == "classification" and not contrastive:
            cnn_flattened_size = 27
        elif self.task == "regression_coords" or self.task == "regression_angles" and not contrastive:
            cnn_flattened_size = 3
        elif contrastive:
            cnn_flattened_size = 256
        
        # The architecture is different depending on whether we want to include the previous predictions or not
        if self.previous:

            # Define the combination MLP
            self.combination_mlp = TwoInputMLP(previous_predictions_size=self.previous_predictions_size, cnn_flattened_size=cnn_flattened_size, 
                                            neurons=self.neurons, output_size=self.output_size, task=self.task)
            
            # Define the final activation depending on the task
            if self.task == "classification" and not contrastive:
                self.final_activation = nn.LogSoftmax(dim=1)
            elif (self.task == "regression_angles" or self.task == "regression_coords") and (not contrastive):
                self.final_activation = nn.Sigmoid()
            elif contrastive:
                self.final_activation = None

    def forward(self, x, previous_predictions, original_shapes):

        # Get the batch size
        batch_size = x.size(0)

        # For each block
        for i in range(self.n_blocks):
                
            # Apply the depthwise separable convolution layers
            x = self.depthwise_conv(x)

            # Apply attention module
            x = self.attention_module(x)

        # Apply global average pooling
        x = self.global_avg_pool(x)

        # Reshape the output of the depthwise separable convolution layers
        x = x.view(batch_size, -1)

        # Apply the final fully connected layers
        x = F.relu(self.final_linear(x))

        # If we do want to include the previous predictions, then we do the following
        if self.previous:

            # Pass the previous predictions through the combination MLP
            x = self.combination_mlp(previous_predictions, x)

            # Apply the final activation if it is not none
            if self.final_activation is not None:
                x = self.final_activation(x)

            # The output is different, depending on if the task is regression of angles or classification
            if self.task == "regression_angles":
                return torch.round(x * 360, 1)
            elif self.task == "regression_coords":
                # Create tensor with the shapes we want to multiply by
                shapes_tensor = torch.tensor([original_shapes[2], original_shapes[3], original_shapes[4]]).cuda()
                # Multiply the two together
                output_x = x * shapes_tensor
                # Return it rounded to the first decimal point
                return output_x
            else:
                return x
        
        # If we don't want to include the previous predictions, then we just return the output of the CNN
        else:
            return x