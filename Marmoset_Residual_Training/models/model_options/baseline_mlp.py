import torch
import torch.nn as nn
import numpy as np

from ..model_builders.twoinput_mlp import TwoInputMLP

# Two layer MLP that takes in the cube and outputs either the direction, angle,
# or the actual coordinates of the streamline node.
class Baseline_MLP(nn.Module):

    # Constructor
    def __init__(self, cnn_flattened_size, hidden_size, output_size, task="classification"):

        # Inherit from parent
        super(Baseline_MLP, self).__init__()

        # Define the output size (different depending on task)
        self.output_size = output_size

        # Define the shape of the flattened output of the CNN
        self.cnn_flattened_size = cnn_flattened_size

        # Define the number of neurons
        self.neurons = hidden_size

        # Define the input size of the previous predictions MLP - will always be output * 2
        self.previous_predictions_size = self.output_size * 2
        
        # Define the task
        self.task = task
        
        # Inherit the layers from the TwoInputMLP
        # Define the combination MLP
        self.combination_mlp = TwoInputMLP(previous_predictions_size=self.previous_predictions_size, cnn_flattened_size=self.cnn_flattened_size, 
                                           neurons=self.neurons, output_size=self.output_size, task=self.task)
        
        # Define the final activation depending on the task
        if task == "classification":
            self.final_activation = nn.Softmax(dim=0)
        elif task == "regression_angles":
            self.final_activation = nn.Sigmoid()

    # Forward pass
    def forward(self, x, previous_predictions):

        # Pass through the combination MLP
        x = self.combination_mlp(previous_predictions, x)
        
        # Pass through the final activation
        x = self.final_activation(x)
                
        # The output is different, depending on if the task is regression of angles or classification
        if self.task == "regression_angles":
            return np.round(x * 360, 1)
        else:
            return x
