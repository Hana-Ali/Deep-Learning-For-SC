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
    def __init__(self, in_channels, num_rnn_layers, num_rnn_hidden_neurons):

        # Call parent constructor
        super(CNN_Attention, self).__init__()

        # Define self attributes
        self.in_channels = in_channels
        self.num_rnn_layers = num_rnn_layers
        self.num_rnn_hidden_neurons = num_rnn_hidden_neurons

        # Define CNN part
        self.cnn = self.build_convolutional_layers()

        # Define Attention part
        self.attention = self.build_attention_layers()

        # Define the adaptive average pool
        self.adaptive_average_pool = nn.AdaptiveAvgPool3d(1)

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
            nn.ReLU()
        )

        # Define the second convolution block
        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(in_channels=filters[0], out_channels=filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(filters[1]),
            nn.ReLU()
        )

        # Define the third convolution block
        self.conv_block_3 = nn.Sequential(
            nn.Conv3d(in_channels=filters[1], out_channels=filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(filters[2]),
            nn.ReLU()
        )

        # Define the 1x1x1 convolution block
        self.conv_block_4 = nn.Sequential(
            nn.Conv3d(in_channels=filters[2], out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(1),
            nn.ReLU()
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

        # Flatten the input
        flattened_input = input_x.view(input_x.shape[0], -1)

        # Define the filter sizes
        neurons = [512, 256, 128, 3]

        # Define the MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(flattened_input.shape[1], neurons[0]),
            nn.ReLU(),
            nn.Linear(neurons[0], neurons[1]),
            nn.ReLU(),
            nn.Linear(neurons[1], neurons[2]),
            nn.ReLU(),
            nn.Linear(neurons[2], neurons[3])
        )

        # Pass the input through the MLP
        mlp_output = self.mlp(flattened_input)

        # Return the output
        return mlp_output

    
    # Forward pass
    def forward(self, input_x):

        # Since we want to do the convolution on every channel separately, we need to
        # reshape the input, so the channels are stacked in the batch dimension
        # (N, C, H, W) -> (N * C, 1, H, W) - http://bitly.ws/PGMQ
        
        # Do the first block of convolution and attention
        first_block = self.pass_pass_pool_multiply(input_x)

        # Do the second block of convolution and attention
        second_block = self.pass_pass_pool_multiply(first_block)

        # Do the third block of convolution and attention
        third_block = self.pass_pass_pool_multiply(second_block)

        # Do the fourth block of convolution and attention
        fourth_block = self.pass_pass_pool_multiply(third_block)

        # Do the fifth block of convolution and attention
        fifth_block = self.pass_pass_pool_multiply(fourth_block)

        # Do the sixth block of convolution and attention
        sixth_block = self.pass_pass_pool_multiply(fifth_block)

        # Do the final MLP pass
        final_mlp = self.mlp_pass(sixth_block)

        # Print the final MLP shape
        print(final_mlp.shape)

        return final_mlp





