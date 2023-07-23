<<<<<<< HEAD
from .encoders import *
from .decoders import *

###############################################################
################## Convolutional Autoencoder ##################
###############################################################
class ConvolutionalAutoEncoder(nn.Module):

    # Constructor
    def __init__(self, input_shape=None, n_features=1, base_width=32, encoder_blocks=None, decoder_blocks=None,
                feature_dilation=2, downsampling_stride=2, interpolation_mode="trilinear",
                encoder_class=MyronenkoEncoder, decoder_class=None, n_outputs=None, layer_widths=None,
                decoder_mirrors_encoder=False, activation=None, use_transposed_convolutions=False, kernel_size=3):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.convolutional_autoencoder = self.build_convolutional_autoencoder(input_shape, n_features, base_width, encoder_blocks, decoder_blocks,
                                                                                feature_dilation, downsampling_stride, interpolation_mode,
                                                                                encoder_class, decoder_class, n_outputs, layer_widths,
                                                                                decoder_mirrors_encoder, activation, use_transposed_convolutions, 
                                                                                kernel_size)

    # Build the convolutional autoencoder
    def build_convolutional_autoencoder(self, input_shape, n_features, base_width, encoder_blocks, decoder_blocks,
                                        feature_dilation, downsampling_stride, interpolation_mode, encoder_class,
                                        decoder_class, n_outputs, layer_widths, decoder_mirrors_encoder, activation,
                                        use_transposed_convolutions, kernel_size):
        
        # If the encoder blocks are not specified, we use the default ones
        if encoder_blocks is None:
            encoder_blocks = [1, 2, 2, 4]

        # Define the base width
        self.base_width = base_width

        # Define the encoder
        self.encoder = encoder_class(num_features=n_features, base_width=base_width, layer_blocks=encoder_blocks,
                                    feature_dilation=feature_dilation, downsampling_stride=downsampling_stride,
                                    layer_widths=layer_widths, kernel_size=kernel_size)
        
        # Get the decoder class and blocks
        decoder_class, decoder_blocks = self.set_decoder_blocks(decoder_class, encoder_blocks, decoder_mirrors_encoder,
                                                        decoder_blocks)

        # Define the decoder
        self.decoder = decoder_class(num_features=n_features, base_width=base_width, layer_blocks=decoder_blocks,
                                    feature_dilation=feature_dilation, downsampling_stride=downsampling_stride,
                                    layer_widths=layer_widths, kernel_size=kernel_size)
        
        # Set the final convolution
        self.set_final_convolution(num_features=n_features)

        # Set the activation
        self.set_activation(activation=activation)

        # Return the convolutional autoencoder
        if self.activation is None:
            return nn.Sequential(self.encoder, self.decoder, self.final_convolution)
        else:
            return nn.Sequential(self.encoder, self.decoder, self.final_convolution, self.activation)

    # Set the decoder blocks
    def set_decoder_blocks(self, decoder_class, encoder_blocks, decoder_mirrors_encoder, decoder_blocks):

        # If the decoder is mirror encoder
        if decoder_mirrors_encoder:
            # The decoder block is the encoder block
            decoder_blocks = encoder_blocks

            # If the decoder class is not specified, we use the MirroredDecoder
            if decoder_class is None:
                decoder_class = MirroredDecoder
            
        # If the deocder blocks is None
        elif decoder_blocks is None:

            # Define it as 1 for every encoder block
            decoder_blocks = [1] * len(encoder_blocks)

            # If the decoder class is not specified, we use the MyronenkoDecoder
            if decoder_class is None:
                decoder_class = MyronenkoDecoder

        # Return the decoder class and blocks
        return decoder_class, decoder_blocks
    
    # Set the final convolution
    def set_final_convolution(self, num_features):
        # Define the final convolution as a 1x1x1 convolution
        self.final_convolution = conv1x1x1(in_channels=self.base_width, out_channels=num_features, stride=1)

    # Set the activation
    def set_activation(self, activation=None):
        # If it's sigmoid
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        # If it's softmax
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        # If it's relu
        elif activation == "relu":
            self.activation = nn.ReLU()
        # If it's leaky relu
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        # If it's none
        else:
            self.activation = None

    # Forward
    def forward(self, x):
        return self.convolutional_autoencoder(x)
        
=======
from .encoders import *
from .decoders import *

###############################################################
################## Convolutional Autoencoder ##################
###############################################################
class ConvolutionalAutoEncoder(nn.Module):

    # Constructor
    def __init__(self, input_shape=None, n_features=1, base_width=32, encoder_blocks=None, decoder_blocks=None,
                feature_dilation=2, downsampling_stride=2, interpolation_mode="trilinear",
                encoder_class=MyronenkoEncoder, decoder_class=None, n_outputs=None, layer_widths=None,
                decoder_mirrors_encoder=False, activation=None, use_transposed_convolutions=False, kernel_size=3):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.convolutional_autoencoder = self.build_convolutional_autoencoder(input_shape, n_features, base_width, encoder_blocks, decoder_blocks,
                                                                                feature_dilation, downsampling_stride, interpolation_mode,
                                                                                encoder_class, decoder_class, n_outputs, layer_widths,
                                                                                decoder_mirrors_encoder, activation, use_transposed_convolutions, 
                                                                                kernel_size)

    # Build the convolutional autoencoder
    def build_convolutional_autoencoder(self, input_shape, n_features, base_width, encoder_blocks, decoder_blocks,
                                        feature_dilation, downsampling_stride, interpolation_mode, encoder_class,
                                        decoder_class, n_outputs, layer_widths, decoder_mirrors_encoder, activation,
                                        use_transposed_convolutions, kernel_size):
        
        # If the encoder blocks are not specified, we use the default ones
        if encoder_blocks is None:
            encoder_blocks = [1, 2, 2, 4]

        # Define the base width
        self.base_width = base_width

        # Define the encoder
        self.encoder = encoder_class(num_features=n_features, base_width=base_width, layer_blocks=encoder_blocks,
                                    feature_dilation=feature_dilation, downsampling_stride=downsampling_stride,
                                    layer_widths=layer_widths, kernel_size=kernel_size)
        
        # Get the decoder class and blocks
        decoder_class, decoder_blocks = self.set_decoder_blocks(decoder_class, encoder_blocks, decoder_mirrors_encoder,
                                                        decoder_blocks)

        # Define the decoder
        self.decoder = decoder_class(num_features=n_features, base_width=base_width, layer_blocks=decoder_blocks,
                                    feature_dilation=feature_dilation, downsampling_stride=downsampling_stride,
                                    layer_widths=layer_widths, kernel_size=kernel_size)
        
        # Set the final convolution
        self.set_final_convolution(num_features=n_features)

        # Set the activation
        self.set_activation(activation=activation)

        # Return the convolutional autoencoder
        if self.activation is None:
            return nn.Sequential(self.encoder, self.decoder, self.final_convolution)
        else:
            return nn.Sequential(self.encoder, self.decoder, self.final_convolution, self.activation)

    # Set the decoder blocks
    def set_decoder_blocks(self, decoder_class, encoder_blocks, decoder_mirrors_encoder, decoder_blocks):

        # If the decoder is mirror encoder
        if decoder_mirrors_encoder:
            # The decoder block is the encoder block
            decoder_blocks = encoder_blocks

            # If the decoder class is not specified, we use the MirroredDecoder
            if decoder_class is None:
                decoder_class = MirroredDecoder
            
        # If the deocder blocks is None
        elif decoder_blocks is None:

            # Define it as 1 for every encoder block
            decoder_blocks = [1] * len(encoder_blocks)

            # If the decoder class is not specified, we use the MyronenkoDecoder
            if decoder_class is None:
                decoder_class = MyronenkoDecoder

        # Return the decoder class and blocks
        return decoder_class, decoder_blocks
    
    # Set the final convolution
    def set_final_convolution(self, num_features):
        # Define the final convolution as a 1x1x1 convolution
        self.final_convolution = conv1x1x1(in_channels=self.base_width, out_channels=num_features, stride=1)

    # Set the activation
    def set_activation(self, activation=None):
        # If it's sigmoid
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        # If it's softmax
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        # If it's relu
        elif activation == "relu":
            self.activation = nn.ReLU()
        # If it's leaky relu
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        # If it's none
        else:
            self.activation = None

    # Forward
    def forward(self, x):
        return self.convolutional_autoencoder(x)
        
>>>>>>> d2d815127215b7b2d0d29b4150a09d943a4f1004
