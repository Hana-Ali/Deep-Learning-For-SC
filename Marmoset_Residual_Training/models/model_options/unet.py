<<<<<<< HEAD
<<<<<<< HEAD
import torch
from models.model_builders import *

###############################################################
######################## U-Net Encoder ########################
###############################################################
class UNetEncoder(MyronenkoEncoder):

    # Define the forward pass
    def forward(self, x_input):

        # Define the outputs
        outputs = []

        # For each layer in the encoder
        for layer, downsampling_convolution in self.myronenko_encoder:
            
            # Pass the input through the layer
            x_input = layer(x_input)

            # Insert the output into the outputs
            outputs.insert(0, x_input)

            # If the downsampling convolution is not None, then we do 1x1x1 convolution
            if downsampling_convolution is not None:
                x_input = downsampling_convolution(x_input)

        # Add the last layer to the outputs
        x_input = self.layers[-1](x_input)
        outputs.insert(0, x_input)

        # Return the outputs
        return outputs


###############################################################
######################## U-Net Decoder ########################
###############################################################
class UNetDecoder(MirroredDecoder):

    # Calculate the layer widths
    def calculate_layer_widths(self, depth):

        # Get them from MirroredDecoder
        in_width, out_width = super().calculate_layer_widths(depth=depth)

        # Id the deoth is not at the last block
        if depth != len(self.layer_blocks) - 1:

            # Double the in width
            in_width *= 2

        # Print out the layer
        print("Decoder Layer {}:".format(depth), in_width, out_width)

        # Return the in and out width
        return in_width, out_width
    
    # Define the forward pass
    def forward(self, x_input):

        # x is the first input
        x = x_input[0]

        # For each pre upsampling convolution, upsampling convolution, and layer
        for index, (pre_upsampling_convolution, upsampling_convolution, layer) in enumerate(self.mirrored_decoder):

            # Pass the input through the layer
            x = layer(x)

            # Pass the input through the pre upsampling convolution
            x = pre_upsampling_convolution(x)

            # Pass the input through the upsampling convolution
            x = upsampling_convolution(x)

            # Concatenate
            x = torch.cat((x, x_input[index + 1]), dim=1)

        # Pass the input through the last layer
        x = self.layers[-1](x)

        # Return the output
        return x

###############################################################
######################## U-Net General ########################
###############################################################
class UNet(ConvolutionalAutoEncoder):

    # Constructor
    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetDecoder, n_outputs=1, **kwargs):
        super().__init__(*args, encoder_class=encoder_class, decoder_class=decoder_class, n_outputs=n_outputs, **kwargs)
=======
=======
>>>>>>> d2d815127215b7b2d0d29b4150a09d943a4f1004
import torch
from models.model_builders import *

###############################################################
######################## U-Net Encoder ########################
###############################################################
class UNetEncoder(MyronenkoEncoder):

    # Define the forward pass
    def forward(self, x_input):

        # Define the outputs
        outputs = []

        # For each layer in the encoder
        for layer, downsampling_convolution in self.myronenko_encoder:
            
            # Pass the input through the layer
            x_input = layer(x_input)

            # Insert the output into the outputs
            outputs.insert(0, x_input)

            # If the downsampling convolution is not None, then we do 1x1x1 convolution
            if downsampling_convolution is not None:
                x_input = downsampling_convolution(x_input)

        # Add the last layer to the outputs
        x_input = self.layers[-1](x_input)
        outputs.insert(0, x_input)

        # Return the outputs
        return outputs


###############################################################
######################## U-Net Decoder ########################
###############################################################
class UNetDecoder(MirroredDecoder):

    # Calculate the layer widths
    def calculate_layer_widths(self, depth):

        # Get them from MirroredDecoder
        in_width, out_width = super().calculate_layer_widths(depth=depth)

        # Id the deoth is not at the last block
        if depth != len(self.layer_blocks) - 1:

            # Double the in width
            in_width *= 2

        # Print out the layer
        print("Decoder Layer {}:".format(depth), in_width, out_width)

        # Return the in and out width
        return in_width, out_width
    
    # Define the forward pass
    def forward(self, x_input):

        # x is the first input
        x = x_input[0]

        # For each pre upsampling convolution, upsampling convolution, and layer
        for index, (pre_upsampling_convolution, upsampling_convolution, layer) in enumerate(self.mirrored_decoder):

            # Pass the input through the layer
            x = layer(x)

            # Pass the input through the pre upsampling convolution
            x = pre_upsampling_convolution(x)

            # Pass the input through the upsampling convolution
            x = upsampling_convolution(x)

            # Concatenate
            x = torch.cat((x, x_input[index + 1]), dim=1)

        # Pass the input through the last layer
        x = self.layers[-1](x)

        # Return the output
        return x

###############################################################
######################## U-Net General ########################
###############################################################
class UNet(ConvolutionalAutoEncoder):

    # Constructor
    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetDecoder, n_outputs=1, **kwargs):
        super().__init__(*args, encoder_class=encoder_class, decoder_class=decoder_class, n_outputs=n_outputs, **kwargs)
<<<<<<< HEAD
>>>>>>> d2d815127215b7b2d0d29b4150a09d943a4f1004
=======
>>>>>>> d2d815127215b7b2d0d29b4150a09d943a4f1004
        self.set_final_convolution(n_outputs=n_outputs)