import torch
import torch.nn as nn
from ..model_builders.twoinput_mlp import TwoInputMLP

class voxelwise_MLP(nn.Module):
    def __init__(self, channels, task="classification", previous=True, output_size=1):
        super(voxelwise_MLP, self).__init__()
        
        # Define attributes
        self.task = task
        self.previous = previous
        self.output_size = output_size

        # Define hidden layer sizes
        hidden_layers = [128, 256, 512, 256, 128, 64, 16, 8]

        # Assert that the number of classes matches the task
        if self.task == "classification":
            assert self.output_size == 27, "Output size must be 27 for classification task"
        elif self.task == "regression_angles":
            assert self.output_size == 3, "Output size must be 3 for regression angles task"
        elif self.task == "regression_coords":
            assert self.output_size == 3, "Output size must be 3 for regression coordinates task"
        else:
            raise NotImplementedError("Task {} is not implemented".format(self.task))


        # Create a list to hold the layers
        layers = []

        # Input layer
        layers.append(nn.Linear(channels, hidden_layers[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())

        # Output layer - to change to [batchsize, 1] change out_features to 1
        layers.append(nn.Linear(hidden_layers[-1], self.output_size))

        # Define the final activation depending on the task
        if self.task == "classification":
            layers.append(nn.LogSoftmax(dim=1))
        elif self.task == "regression_angles" or self.task == "regression_coords":
            layers.append(nn.Sigmoid())

        # Combine the layers
        self.network = nn.Sequential(*layers)

    def forward(self, x, previous_predictions, original_shapes):

        # Pass through the network
        x = self.network(x)

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