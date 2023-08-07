import torch
import torch.nn as nn

# Define class that takes in two inputs, the first the output of the efficient net, and the
# second the previous predictions, and outputs the final predictions as a combination of both
class TwoInputMLP(nn.Module):
    def __init__(self, previous_predictions_size, cnn_flattened_size, neurons, output_size, 
                 task="classification", batch_norm=True):
        
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

        self.prev_pred_FC = [nn.Linear(previous_predictions_size, neurons),  # First MLP for input of size [batch_size, 6]
                            nn.ReLU(inplace=True)]
        if batch_norm:
            self.prev_pred_FC.append(nn.BatchNorm1d(neurons))
        else:
            self.prev_pred_FC.append(nn.GroupNorm(1, neurons))
        self.prev_pred_FC = nn.Sequential(*self.prev_pred_FC)

        self.cnn_output_FC = [nn.Linear(cnn_flattened_size, neurons),  # Second MLP for input of size [batch_size, 3]
                              nn.ReLU(inplace=True)]
        if batch_norm:
            self.cnn_output_FC.append(nn.BatchNorm1d(neurons))
        else:
            self.cnn_output_FC.append(nn.GroupNorm(1, neurons))
        self.cnn_output_FC = nn.Sequential(*self.cnn_output_FC)

        self.combo_FC = [nn.Linear(neurons * 2, output_size),  # Output layer
                            nn.ReLU(inplace=True)]
        if batch_norm:
            self.combo_FC.append(nn.BatchNorm1d(output_size))
        else:
            self.combo_FC.append(nn.GroupNorm(1, output_size))
        self.combo_FC = nn.Sequential(*self.combo_FC)


    def forward(self, previous_predictions, efficientnet_output):

        # Flatten the inputs along the second and third dimensions
        previous_predictions = previous_predictions.view(previous_predictions.size(0), -1)
        efficientnet_output = efficientnet_output.view(efficientnet_output.size(0), -1)

        # Pass each input through their respective MLPs
        previous_predictions = self.prev_pred_FC(previous_predictions)
        efficientnet_output = self.cnn_output_FC(efficientnet_output)
        
        # Concatenate the outputs of both MLPs along the last dimension
        x = torch.cat((previous_predictions, efficientnet_output), dim=-1)

        # Pass the concatenated output through the final layer
        x = self.combo_FC(x)        

        return x