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

##############################################################
###################### CNN_BiLSTM Block ######################
##############################################################
class CNN_BiLSTM(nn.Module):

    # Constructor
    def __init__(self, in_channels, num_rnn_layers, num_rnn_hidden_neurons):

        # Call parent constructor
        super(CNN_BiLSTM, self).__init__()

        # Define self attributes
        self.in_channels = in_channels
        self.num_rnn_layers = num_rnn_layers
        self.num_rnn_hidden_neurons = num_rnn_hidden_neurons

        # Define CNN part
        self.cnn = self.build_convolutional_layers()

        # Define Attention part
        self.attention = self.build_attention_layers()

        # Define BiLSTM part
        self.bilstm = self.build_bilstm_layers(num_rnn_layers, num_rnn_hidden_neurons)

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
    def build_attention_layers(self, num_features, reduction):

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

    # Build BiLSTM layers
    def build_bilstm_layers(self, input_size, num_rnn_layers, num_rnn_hidden_neurons):

        # Define the BiLSTM layers
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=num_rnn_hidden_neurons,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True
        )

        # Return the BiLSTM layers
        return self.bilstm


    # Forward pass
    def forward(self, x):

        # Since we want to do the convolution on every channel separately, we need to
        # reshape the input, so the channels are stacked in the batch dimension
        # (N, C, H, W) -> (N * C, 1, H, W) - http://bitly.ws/PGMQ
        pass


