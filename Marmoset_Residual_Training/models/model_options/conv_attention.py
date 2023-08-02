import torch
import torch.nn as nn

# https://github.com/dalinvip/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch/tree/master/models
# A convolutional recurrent neural network with attention framework for speech separation in monaural recordings
# https://github.com/hujie-frank/SENet
# https://github.com/crownpku/Information-Extraction-Chinese/tree/master/NER_IDCNN_CRF
# Robust Object Classification Approach using Spherical Harmonics
# https://github.com/fyancy/SSMN/blob/main/SSMN_rev02/my_utils/Attention_Block.py
# Semi-supervised meta-learning networks with squeeze-and-excitation attention for few-shot fault diagnosis
# https://github.com/tsmotlp/Attentions-on-Image/blob/master/se_block.py

#############################################################
#################### CNN_Attention Block ####################
#############################################################
class CNN_Attention(nn.Module):

    # Constructor
    def __init__(self, in_channels=45, num_rnn_layers=2, num_rnn_hidden_neurons=1000, cube_size=15, num_nodes=40, num_coordinates=3, 
                 prev_output_size=32, combination=True):

        # Call parent constructor
        super(CNN_Attention, self).__init__()

        # Define self attributes
        self.in_channels = in_channels
        self.num_rnn_layers = num_rnn_layers
        self.num_rnn_hidden_neurons = num_rnn_hidden_neurons

        # Define the number of nodes and coordinates / node
        self.num_nodes = num_nodes
        self.num_coordinates = num_coordinates

        # Define whether we want to use the previous predictions (combine or not)
        self.combination = combination

        # Define the output size of the previous predictions MLP
        self.previous_predictions_mlp_output_size = prev_output_size

        # Define the flattened input size to the CNN/Attention MLP
        self.flattened_input_size = cube_size * cube_size * cube_size * self.in_channels

        # Define CNN part
        self.cnn = self.build_convolutional_layers()

        # Define Attention part
        self.attention = self.build_attention_layers()

        # Define the adaptive average pool
        self.adaptive_average_pool = nn.AdaptiveAvgPool3d(1)

        # Define the previous predictions MLP
        self.previous_predictions_mlp = Previous_Predictions_MLP(num_coordinates=self.num_coordinates, num_nodes=self.num_nodes,
                                                                 output_size=self.previous_predictions_mlp_output_size)
        
        # Define the combination MLP
        self.combination_mlp = Combination_MLP(num_coordinates=self.num_coordinates, num_nodes=self.num_nodes,
                                                  output_size=self.previous_predictions_mlp_output_size, flattened_input_size=self.flattened_input_size)

        # Define the MLP part if we don't want to use the previous predictions
        self.pointmap_mlp = self.build_pointmap_mlp_layers()

    # Build convolutional layers
    def build_convolutional_layers(self):

        # Since we want to do the convolution on every channel separately, we need to
        # reshape the input, so the channels are stacked in the batch dimension
        # (N, C, H, W) -> (N * C, 1, H, W) - http://bitly.ws/PGMQ
        
        # Define the filter sizes
        filters = [32, 64, 128]

        # Define the first convolution block
        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(inplace=True)
        )

        # Define the second convolution block
        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(in_channels=filters[0], out_channels=filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(filters[1]),
            nn.ReLU(inplace=True)
        )

        # Define the third convolution block
        self.conv_block_3 = nn.Sequential(
            nn.Conv3d(in_channels=filters[1], out_channels=filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(filters[2]),
            nn.ReLU(inplace=True)
        )

        # Define the 1x1x1 convolution block
        self.conv_block_4 = nn.Sequential(
            nn.Conv3d(in_channels=filters[2], out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )

        # Return the convolutional layers
        return nn.Sequential(self.conv_block_1, self.conv_block_2, self.conv_block_3, self.conv_block_4)

    # Build attention layers
    def build_attention_layers(self, num_features=1, reduction=4):

        # Build the attention layers from SE-Net
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(num_features, num_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction, num_features, bias=False),
            nn.Sigmoid()
        )

        # Return the attention layers
        return nn.Sequential(self.squeeze, self.excitation)
    
    # Build the pointmap MLP layers
    def build_pointmap_mlp_layers(self):

        # Define the filter sizes
        neurons = [512, 256, 128]

        # Define the MLP layers
        self.pointmap_mlp = nn.Sequential(
            nn.Linear(self.flattened_input_size, neurons[0]),
            nn.ReLU(inplace=True),
            nn.Linear(neurons[0], neurons[1]),
            nn.ReLU(inplace=True),
            nn.Linear(neurons[1], neurons[2]),
            nn.ReLU(inplace=True),
            nn.Linear(neurons[2], self.num_nodes * self.num_coordinates)
        )

        # Return the MLP layers
        return self.pointmap_mlp
    
    # Define the convolution and attention pass
    def convolution_attention_pass(self, input_x):

        # Since we want to do the convolution on every channel separately, we need to
        # reshape the input, so the channels are stacked in the batch dimension
        # (N, C, H, W) -> (N * C, 1, H, W) - http://bitly.ws/PGMQ

        # Reshape the input
        reshaped_input = input_x.view(-1, 1, input_x.shape[2], input_x.shape[3], input_x.shape[4])

        # Pass the input through the convolutional layers
        convolution_output = self.cnn(reshaped_input)

        # Reshape the output
        reshaped_output = convolution_output.view(input_x.shape[0], input_x.shape[1], convolution_output.shape[2], convolution_output.shape[3], convolution_output.shape[4])

        # Pass the input through the attention layers
        attention_output = self.attention(reshaped_output)

        # Multiply the attention output with the convolution output
        multiplied_output = attention_output * reshaped_output

        # Return the output
        return multiplied_output
    
    # Define the pass -> pass -> pool -> multiply function
    def pass_pass_pool_multiply(self, input_x):

        # Get the first pass through the convolution and attention layers
        first_pass = self.convolution_attention_pass(input_x)

        # Get the second pass through the convolution and attention layers
        second_pass = self.convolution_attention_pass(first_pass)

        # Average pool result of the second pass
        average_pool = self.adaptive_average_pool(second_pass)

        # Multiply the average pool with the first pass
        multiplied_average_pool = average_pool * input_x

        # Return the multiplied average pool
        return multiplied_average_pool
        
    # Define the MLP pass
    def mlp_pass(self, input_x):

        # Calculate the flattened input size
        flattened_input_size = input_x.shape[1] * input_x.shape[2] * input_x.shape[3] * input_x.shape[4]

        # Flatten the input, preserving batch size
        flattened_input = input_x.view(-1, flattened_input_size)

        # Pass the input through the MLP
        mlp_output = self.pointmap_mlp(flattened_input)

        # Reshape the output
        mlp_output = mlp_output.view(-1, self.num_nodes, self.num_coordinates)

        # Return the output
        return mlp_output

    
    # Forward pass
    def forward(self, wmfod_input, previous_predictions):

        # Since we want to do the convolution on every channel separately, we need to
        # reshape the input, so the channels are stacked in the batch dimension
        # (N, C, H, W) -> (N * C, 1, H, W) - http://bitly.ws/PGMQ
        
        # Do the first block of convolution and attention
        first_block = self.pass_pass_pool_multiply(wmfod_input)
        print("Shape of first_block", first_block.shape)

        # Do the second block of convolution and attention
        second_block = self.pass_pass_pool_multiply(first_block)
        print("Shape of second_block", second_block.shape)

        # Do the third block of convolution and attention
        third_block = self.pass_pass_pool_multiply(second_block)
        print("Shape of third_block", third_block.shape)

        # Do the fourth block of convolution and attention
        fourth_block = self.pass_pass_pool_multiply(third_block)
        print("Shape of fourth_block", fourth_block.shape)

        # Do the fifth block of convolution and attention
        fifth_block = self.pass_pass_pool_multiply(fourth_block)
        print("Shape of fifth_block", fifth_block.shape)

        # Do the sixth block of convolution and attention
        sixth_block = self.pass_pass_pool_multiply(fifth_block)
        print("Shape of sixth_block", sixth_block.shape)

        # If we want to use the previous predictions
        if self.combination:

            # Pass the previous predictions through the MLP
            previous_predictions_mlp_output = self.previous_predictions_mlp(previous_predictions)
            print("Shape of previous_predictions_mlp_output", previous_predictions_mlp_output.shape)

            # Pass the previous predictions and the sixth block through the combination MLP
            final_mlp = self.combination_mlp(previous_predictions_mlp_output, sixth_block)
            print("Shape of final_mlp", final_mlp.shape)

        # If we don't want to use the previous predictions
        else:
            # Do the final MLP pass
            final_mlp = self.mlp_pass(sixth_block)
            print("Shape of final_mlp", final_mlp.shape)

        return final_mlp

# Define the Previous Predictions MLP as a class with a forward pass
class Previous_Predictions_MLP(nn.Module):

    # Constructor
    def __init__(self, num_coordinates=3, num_nodes=40, output_size=32):

        # Call parent constructor
        super(Previous_Predictions_MLP, self).__init__()

        # Define the number of nodes and coordinates / node
        self.num_nodes = num_nodes
        self.num_coordinates = num_coordinates

        # Define the output size of the previous predictions MLP
        self.previous_predictions_mlp_output_size = output_size

        # Define the MLP part if we don't want to use the previous predictions
        self.previous_predictions_mlp = self.build_previous_predictions_mlp()

    # Build the first MLP (for the previous predictions)
    def build_previous_predictions_mlp(self):

        # Define the number of neurons in each layer
        neurons = [64]

        # Define the MLP layers
        self.previous_predictions_mlp = nn.Sequential(
            nn.Linear(self.num_coordinates * 2, neurons[0]),
            nn.ReLU(inplace=True),
            nn.Linear(neurons[0], self.previous_predictions_mlp_output_size),
            nn.ReLU(inplace=True)
        )

        # Return the MLP layers
        return self.previous_predictions_mlp
    
    # Forward pass
    def forward(self, previous_predictions):

        # Pass the previous predictions through the MLP
        previous_predictions_mlp_output = self.previous_predictions_mlp(previous_predictions)

        # Reshape the output
        previous_predictions_mlp_output = previous_predictions_mlp_output.view(-1, self.previous_predictions_mlp_output_size)

        # Return the output
        return previous_predictions_mlp_output
    
# Define the Combination MLP as a class with a forward pass
class Combination_MLP(nn.Module):

    # Constructor
    def __init__(self, num_coordinates=3, num_nodes=40, output_size=32, flattened_input_size=27000):

        # Call parent constructor
        super(Combination_MLP, self).__init__()

        # Define the number of nodes and coordinates / node
        self.num_nodes = num_nodes
        self.num_coordinates = num_coordinates

        # Define the output size of the previous predictions MLP
        self.previous_predictions_mlp_output_size = output_size

        # Define the flattened input size to the CNN/Attention MLP
        self.flattened_input_size = flattened_input_size

        # Define the MLP part if we don't want to use the previous predictions
        self.combination_mlp = self.build_combination_mlp()

    # Build the combination MLP
    def build_combination_mlp(self):

        # Define the number of neurons in each layer
        neurons = [512, 256, 128, 64]

        # Define the MLP layers
        self.combination_mlp = nn.Sequential(
            nn.Linear(self.previous_predictions_mlp_output_size + self.flattened_input_size, neurons[0]),
            nn.ReLU(inplace=True),
            nn.Linear(neurons[0], neurons[1]),
            nn.ReLU(inplace=True),
            nn.Linear(neurons[1], neurons[2]),
            nn.ReLU(inplace=True),
            nn.Linear(neurons[2], neurons[3]),
            nn.ReLU(inplace=True),
            nn.Linear(neurons[3], self.num_nodes * self.num_coordinates)
        )

        # Return the MLP layers
        return self.combination_mlp

    # Forward pass
    def forward(self, previous_predictions, cnn_attention_output):
        
        print("Shape of previous_predictions", previous_predictions.shape)
        print("Shape of cnn_attention_output", cnn_attention_output.shape)

        # Flatten the input, preserving batch size
        flattened_input = cnn_attention_output.view(-1, self.flattened_input_size)
        print("Shape of flattened_input", flattened_input.shape)

        # Concatenate the previous predictions with the flattened input
        combination_mlp_input = torch.cat((previous_predictions, flattened_input), dim=1)

        # Pass the input through the MLP
        combination_mlp_output = self.combination_mlp(combination_mlp_input)

        # Reshape the output
        combination_mlp_output = combination_mlp_output.view(-1, self.num_nodes, self.num_coordinates)

        # Return the output
        return combination_mlp_output