from .encoders import *
from .decoders import *

###############################################################
################## Convolutional Autoencoder ##################
###############################################################
class ConvolutionalAutoEncoder(nn.Module):

    # Constructor
    def __init__(self, input_shape=None, in_channels=3, ngf=64, encoder_blocks=None, decoder_blocks=None,
                feature_dilation=2, downsampling_stride=2, interpolation_mode="trilinear",
                encoder_class=MyronenkoEncoder, decoder_class=None, output_channels=None, layer_widths=None,
                decoder_mirrors_encoder=False, activation=None, use_transposed_convolutions=False, kernel_size=3,
                voxel_wise=False):
        super(ConvolutionalAutoEncoder, self).__init__()
        self.convolutional_autoencoder = self.build_convolutional_autoencoder(input_shape, in_channels, ngf, encoder_blocks, decoder_blocks,
                                                                                feature_dilation, downsampling_stride, interpolation_mode,
                                                                                encoder_class, decoder_class, output_channels, layer_widths,
                                                                                decoder_mirrors_encoder, activation, use_transposed_convolutions, 
                                                                                kernel_size, voxel_wise)

    # Build the convolutional autoencoder
    def build_convolutional_autoencoder(self, input_shape, in_channels, ngf, encoder_blocks, decoder_blocks,
                                        feature_dilation, downsampling_stride, interpolation_mode, encoder_class,
                                        decoder_class, output_channels, layer_widths, decoder_mirrors_encoder, activation,
                                        use_transposed_convolutions, kernel_size, voxel_wise):
        
        # If the encoder blocks are not specified, we use the default ones
        if encoder_blocks is None:
            encoder_blocks = [1, 2, 2, 4]

        # Define the attributes of the model
        self.ngf = ngf
        self.out_channels = output_channels
        self.number_downsampling = 2
        self.norm_layer = nn.BatchNorm3d

        # Define whether it's voxel_wise or not
        self.voxel_wise = voxel_wise

        # Define the image and non-image models
        self.non_img_model = self.define_non_img_model()
        self.joint_model = self.define_joint_model()

        # Define the encoder
        self.encoder = encoder_class(in_channels=in_channels, ngf=ngf, block_layers=encoder_blocks,
                                    feature_dilation=feature_dilation, downsampling_stride=downsampling_stride,
                                    layer_widths=layer_widths, kernel_size=kernel_size)
        
        # Get the decoder class and blocks
        decoder_class, decoder_blocks = self.set_decoder_blocks(decoder_class, encoder_blocks, decoder_mirrors_encoder,
                                                                decoder_blocks)

        # Define the decoder
        self.decoder = decoder_class(ngf=ngf, block_layers=decoder_blocks,
                                     upsampling_scale=downsampling_stride, feature_reduction_scale=feature_dilation,
                                     upsampling_mode=interpolation_mode, layer_widths=layer_widths,
                                     use_transposed_convolutions=use_transposed_convolutions,
                                     kernel_size=kernel_size)
        
        # Set the final convolution
        self.set_final_convolution(output_channels=output_channels)

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
    
     # Define the processing for the non-image inputs
    def define_non_img_model(self):
        
        # Stores the model
        model = []
        
        # Add convolutions for the injection centers and image coordinates - expected to have self.output_nc channels
        for i in range(self.number_downsampling):
            model += [nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                      self.norm_layer(self.out_channels), 
                          nn.ReLU(True)]
            
        # Return the model
        return nn.Sequential(*model)
            
    # Define joint processing for everything
    def define_joint_model(self):
        
        # Stores the model
        model = []
        
        # Define the factor we multiply by, based on voxel_wise
        if self.voxel_wise:
            factor = 1
        else:
            factor = 3
                    
        # Add final convolutions for image and non-image data
        # Cube output: self.out_channels * 3 channels | Voxel output: self.out_channels channels
        for i in range(self.number_downsampling):
            model += [nn.Conv3d(self.out_channels * factor, self.out_channels * factor, kernel_size=3, stride=1, padding=1, 
                                bias=False),
                          nn.ReLU(True)]
            
        # Final convolution to make the number of channels 1
        # Cube output: self.out_channels * 3 channels | Voxel output: self.out_channels channels
        model += [nn.Conv3d(self.out_channels * factor, 1, kernel_size=3, stride=1, padding=1, bias=False)]
        
        # Cube output: No Adaptive layer | Voxel output: Adaptive layer
        if self.voxel_wise:
            model += [nn.AdaptiveAvgPool3d((1, 1, 1))]
            
        # Return the model
        return nn.Sequential(*model)
    
    # Set the final convolution
    def set_final_convolution(self, output_channels):

        # Depending on if it's voxel_wise or not, the final convolution will either just
        # 1. Make # channels to 1
        # 2. Make # channels AND spatial size to 1

        # Cube output: No Adaptive layer | Voxel output: Adaptive layer

        # If it's voxel_wise
        if self.voxel_wise:
            self.final_convolution = nn.Sequential(
                conv1x1x1(in_channels=self.ngf, out_channels=output_channels, stride=2),
                nn.AdaptiveAvgPool3d((1, 1, 1))
            )
        else:
            self.final_convolution = conv1x1x1(in_channels=self.ngf, out_channels=output_channels, stride=2)

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
    def forward(self, input_x, injection_center, image_coordinates):

        # Define the dimension we concatenate along, depending on voxel wise
        if self.voxel_wise:
            dim = 4
        else:
            dim = 1

        # Do all the convolutions for the b0 first
        input_x = self.convolutional_autoencoder(input_x)

        # Do the convolutional layers for the injection center
        injection_center = self.non_img_model(injection_center)
        
        # Do the convolutional layers for the image coordinates
        image_coordinates = self.non_img_model(image_coordinates)
        
        # Concatenate the data along the number of channels
        # Cube output: Dimension 1 | Voxel output: Dimension 4
        input_x = torch.cat((input_x, injection_center), dim=dim)
        input_x = torch.cat((input_x, image_coordinates), dim=dim)
        
        # Do the joint processing
        joint_data = self.joint_model(input_x)

        # Return the model
        return joint_data