import torch
import torch.nn as nn
import numpy as np

from ..model_builders.twoinput_mlp import TwoInputMLP

# Two layer MLP that takes in the cube and outputs either the direction, angle,
# or the actual coordinates of the streamline node.
class Baseline_MLP(nn.Module):

    # Constructor
    def __init__(self, cnn_flattened_size, hidden_size, output_size, task="classification", contrastive=False):

        # Inherit from parent
        super(Baseline_MLP, self).__init__()
        
        # Define the task
        self.task = task

        # Define the output size (different depending on task)
        self.output_size = output_size

        # Define the number of neurons
        self.neurons = hidden_size
        
        # Define the flattened size
        self.cnn_flattened_size = cnn_flattened_size
        
        # print("cnn flattened size in baseline mlp is", cnn_flattened_size)

        # Define the input size of the previous predictions MLP - will always be output * 2
        self.previous_predictions_size = self.output_size * 2
        
        # Define the task
        self.task = task
        
        # The flattened size and final activation depends on the task
        if self.task == "classification" and not contrastive:
            self.final_activation = nn.LogSoftmax(dim=1)
        elif self.task == "regression_coords" or self.task == "regression_angles" and not contrastive:
            self.final_activation = nn.Sigmoid()
        elif contrastive:
            self.final_activation = None
        
        # Define the combination MLP
        self.combination_mlp = TwoInputMLP(previous_predictions_size=self.previous_predictions_size, cnn_flattened_size=self.cnn_flattened_size, 
                                           neurons=self.neurons, output_size=self.output_size, task=self.task)

    # Forward pass
    def forward(self, x, previous_predictions, original_shapes):
        
        # print("Shape of x", x.shape)
        # print("Shape of previous predictions", previous_predictions.shape)

        # Pass through the combination MLP
        x = self.combination_mlp(previous_predictions, x)
        
        # print("Shape of x right before activation", x.shape)
        # print("x right before activation", x)
        
        # Apply the final activation
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
            # return torch.round(output_x, decimals=1)
            return output_x
        else:
            return x
