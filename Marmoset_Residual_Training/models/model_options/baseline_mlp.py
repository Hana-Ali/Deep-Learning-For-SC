import torch
import torch.nn as nn
import numpy as np

from ..model_builders.efficientnet_utils import TwoInputMLP

# Two layer MLP that takes in the cube and outputs either the direction, angle,
# or the actual coordinates of the streamline node.
class Baseline_MLP(nn.Module):

    # Constructor
    def __init__(self, previous_predictions_size, efficientnet_output_size, hidden_size, output_size, task="classification"):

        # Inherit from parent
        super(Baseline_MLP, self).__init__()

        # Define the number of coordinates (output size)
        self.output_size = output_size

        # Define the input size of efficientnet
        self.efficientnet_output_size = efficientnet_output_size

        # Define the number of neurons
        self.neurons = hidden_size

        # Define the input size of the previous predictions MLP
        self.previous_predictions_size = previous_predictions_size
        
        # Define the task
        self.task = task

        # Inherit the layers from the TwoInputMLP
        # Define the combination MLP
        self.combination_mlp = TwoInputMLP(previous_predictions_size=self.previous_predictions_size, efficientnet_output_size=self.efficientnet_output_size, 
                                           neurons=self.neurons, output_size=self.output_size, task=self.task)
        
        # Define the final activation depending on the task
        if task == "classification":
            self.final_activation = nn.Softmax(dim=1)
        elif task == "regression":
            self.final_activation = nn.Sigmoid()

    # Forward pass
    def forward(self, x, previous_predictions):

        # Pass through the combination MLP
        x = self.combination_mlp(previous_predictions, x)

        # Pass through the final activation
        x = self.final_activation(x)
        
        # The output is different, depending on if the task is regression or classification
        if self.task == "regression":
            return x * 360
        else:
            return x
