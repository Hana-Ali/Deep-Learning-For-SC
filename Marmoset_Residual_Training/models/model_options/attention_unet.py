import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_builders import *


class Attention_UNet(nn.Module):

    def __init__(self, feature_scale=4, in_channels=1, n_classes=3, is_deconv=True,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True,
                 voxel_wise=False, cube_size=16):
        super(Attention_UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes
        self.voxel_wise = voxel_wise
        self.cube_size = cube_size

        # Define the normalization layer
        if self.is_batchnorm:
            self.norm_layer = nn.BatchNorm3d
        else:
            self.norm_layer = nn.InstanceNorm3d

        # Number of downsamplings in the model
        self.number_downsampling = 2

        filters = [64, 128, 256, 512, 1024]
        filters = [int(filt / self.feature_scale) for filt in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        if self.cube_size * 2 < 32:
            self.final = nn.Conv3d(n_classes*4, n_classes, 1, stride=4)
        else:
            self.final = nn.Conv3d(n_classes*4, n_classes, 1, stride=2)

        # Define the image and non-image models
        self.non_img_model = self.define_non_img_model()
        self.joint_model = self.define_joint_model()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

     # Define the processing for the non-image inputs
    def define_non_img_model(self):
        
        # Stores the model
        model = []

        # Add convolutions for the injection centers and image coordinates - expected to have self.output_nc channels
        for i in range(self.number_downsampling):
            model += [nn.Conv3d(self.n_classes, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False),
                      self.norm_layer(self.n_classes), 
                          nn.ReLU(True)]
        
         # final conv (without any concat)
        if self.cube_size * 2 < 32:
            model += [nn.Conv3d(self.n_classes, self.n_classes, kernel_size=3, stride=2,
                               padding=1)]
            
        # Return the model
        return nn.Sequential(*model)
            
    # Define joint processing for everything
    def define_joint_model(self):
        
        # Stores the model
        model = []
                    
        # Add final convolutions for image and non-image data
        # Cube output: self.n_channels * 3 channels | Voxel output: self.n_channels channels
        for i in range(self.number_downsampling):
            model += [nn.Conv3d(self.n_classes * 3, self.n_classes * 3, kernel_size=3, stride=1, padding=1, 
                                bias=False),
                        self.norm_layer(self.n_classes * 3),
                          nn.ReLU(True)]
            
        # Final convolution to make the number of channels 1
        # Cube output: self.n_channels * 3 channels | Voxel output: self.n_channels channels
        model += [nn.Conv3d(self.n_classes * 3, 1, kernel_size=3, stride=1, padding=1, bias=False)]
        
        # Cube output: No Adaptive layer | Voxel output: Adaptive layer
        if self.voxel_wise:
            model += [nn.AdaptiveAvgPool3d((1, 1, 1))]
            
        # Return the model
        return nn.Sequential(*model)

    def forward(self, inputs, injection_centers, image_coordinates):

        if self.voxel_wise:
            dim = 4
        else:
            dim = 1
            
        # Check if the input is too small
        if inputs.shape[2] < 32:
            inputs = nn.Upsample(scale_factor=2, mode='nearest')(inputs)
            injection_centers = nn.Upsample(scale_factor=2, mode='nearest')(injection_centers)
            image_coordinates = nn.Upsample(scale_factor=2, mode='nearest')(image_coordinates)
            
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))

        # Apply convolutions to injection centers and image coordinates
        injection_centers = self.non_img_model(injection_centers)
        image_coordinates = self.non_img_model(image_coordinates)

        # Concatenate the final output with the injection centers and image coordinates
        final = torch.cat((final, injection_centers), dim=dim)
        final = torch.cat((final, image_coordinates), dim=dim)
        
        # Do the joint processing
        final = self.joint_model(final)

        return final
    
    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)