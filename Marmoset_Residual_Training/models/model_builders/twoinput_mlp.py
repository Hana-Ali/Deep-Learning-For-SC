import torch
import torch.nn as nn

# Define class that takes in two inputs, the first the output of the efficient net, and the
# second the previous predictions, and outputs the final predictions as a combination of both
class TwoInputMLP(nn.Module):
    def __init__(self, previous_predictions_size, cnn_flattened_size, neurons, output_size, task="classification"):
        
        # Call the super constructor
        super(TwoInputMLP, self).__init__()

        # Define attributes
        self.previous_predictions_size = previous_predictions_size
        self.cnn_flattened_size = cnn_flattened_size
        self.neurons = neurons
        self.output_size = output_size

        # print("Previous predictions size", self.previous_predictions_size)
        # print("Efficientnet output size", self.efficientnet_output_size)
        # print("Neurons", self.neurons)
        # print("Output size", self.output_size)

        self.prev_pred_FC = nn.Linear(previous_predictions_size, neurons)  # First MLP for input of size [batch_size, 6]
        self.cnn_output_FC = nn.Linear(cnn_flattened_size, neurons)  # Second MLP for input of size [batch_size, 3]
        self.combo_FC = nn.Linear(neurons * 2, output_size)  # Output layer

    def forward(self, previous_predictions, efficientnet_output):

        # Flatten the inputs along the second and third dimensions
        previous_predictions = previous_predictions.view(previous_predictions.size(0), -1)
        efficientnet_output = efficientnet_output.view(efficientnet_output.size(0), -1)

        # print("Previous predictions shape", previous_predictions.shape)
        # print("Efficientnet output shape", efficientnet_output.shape)

        # Pass each input through their respective MLPs
        previous_predictions = self.prev_pred_FC(previous_predictions)
        efficientnet_output = self.cnn_output_FC(efficientnet_output)

        # print("Previous predictions shape", previous_predictions.shape)
        # print("Efficientnet output shape", efficientnet_output.shape)

        # Pass them through ReLU activation
        previous_predictions = torch.relu(previous_predictions)
        efficientnet_output = torch.relu(efficientnet_output)

        # Concatenate the outputs of both MLPs along the last dimension
        x = torch.cat((previous_predictions, efficientnet_output), dim=-1)

        # Pass the concatenated output through the final layer
        x = self.combo_FC(x)
        

        return x