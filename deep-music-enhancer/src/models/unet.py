"""
U-Net model for 1-D signals
"""
import torch.nn as nn
import torch
from utils import conv1d_halve, conv1d_same, PixelUpscale


class Model(nn.Module):

    # default no dropout, no batchnorm, always use padding=0
    def __init__(self, mono=False, scale=2, odd_length=False,
                 pad_type='zero', dropout=0.0, batchnorm=False):
        
        super(Model, self).__init__()

        assert not (batchnorm and dropout)

        self.scale = scale  # for pixel upscale

        # Mono/Stereo (Input layer & Output layer channel size)
        if mono:
            n_input_ch = 1
            n_output_ch = 1
        else:
            n_input_ch = 2
            n_output_ch = 2
        ch_down_net = [n_input_ch, 128, 256, 512, 512]  # channel size of the downsampling network
        ker_down_net = [65, 33, 17, 9]  # kernel size of the downsampling network
        ch_down_bottle = 512  # channel size of the downsampling bottleneck
        ker_down_bottle = 9  # kernel size of the downsampling bottleneck
        ch_up_bottle = 512  # channel size of the upsampling bottleneck
        ker_up_bottle = 9  # kernel size of the upsampling bottleneck
        ch_up_net = [512, 512, 256, 128, n_output_ch]  # channel size of the upsampling network
        ker_up_net = [17, 33, 65, 9]  # kernel size of the upsampling network



        activation = nn.ReLU(inplace=True)  # activation layer

        
        self.down_net = nn.ModuleList()   # initialize empty downsampling network
        # append downsampling blocks to downsampling network
        for i in range(len(ker_down_net)):
            down_block = nn.ModuleList()

            down_block.append(conv1d_halve(ch_down_net[i], ch_down_net[i+1], ker_down_net[i], pad_type=pad_type))  # Convolution layer which halves the input size
            if batchnorm:  # batch normalization after each downsampling convolution layer
                down_block.append(nn.BatchNorm1d(ch_down_net[i+1]))
            down_block.append(activation)

            down_block = nn.Sequential(*down_block)
            self.down_net.append(down_block)

        
        
        # bottleneck doesn't have residual connections
        bottleneck = nn.ModuleList()   # initialize empty bottleneck(s)
        # downsampling bottleneck
        bottleneck.append(conv1d_halve(ch_down_net[-1], ch_down_bottle, ker_down_bottle, pad_type=pad_type))
        if dropout > 0:
            bottleneck.append(nn.Dropout(dropout))
        bottleneck.append(activation)
        # upsampling bottleneck
        bottleneck.append(conv1d_same(ch_down_bottle, ch_up_bottle * scale, ker_up_bottle, pad_type=pad_type))  # scale back by *2
        if dropout > 0:
            bottleneck.append(nn.Dropout(dropout))
        bottleneck.append(activation)
        # upscale
        bottleneck.append(PixelUpscale(self.scale, odd_output=odd_length))
        self.bottleneck = nn.Sequential(*bottleneck)

        
        
        self.up_net = nn.ModuleList()   # initialize empty upsampling network
        for i in range(len(ker_up_net)):

            n_ch_in = ch_up_net[i] * 2     # residual channels will be added
            n_ch_out = ch_up_net[i+1] * self.scale      # for pixel upscaling

            up_block = nn.ModuleList()
            up_block.append(conv1d_same(n_ch_in, n_ch_out, ker_up_net[i], pad_type=pad_type))  # Convolution which does not change input size
            if dropout > 0:  # dropout after each upsampling convolution layer
                up_block.append(nn.Dropout(dropout))
            if i < len(ker_up_net) - 1:     # use activation except last layer
                up_block.append(activation)
            up_block.append(PixelUpscale(self.scale, odd_output=odd_length))  # for the last layer we use PixelUpscale

            up_block = nn.Sequential(*up_block)
            self.up_net.append(up_block)



    # now that we have 3 models: self.down_net, self.bottleneck, self.up_net
    def forward(self, x):

        y = x.clone()  # make a copy of input x, below we process y

        res = []    # save the output of each downsampling block to be later used as input in upsampling

        for down_block in self.down_net:
            y = down_block(y)  # pipe input y through each downsampling block
            res.append(y)  # save output
    
        y = self.bottleneck(y)  # pipe through the whole bottleneck

        for i, up_block in enumerate(self.up_net):
            y = torch.cat((y, res[-i-1]), dim=1)   # concat upsampling input along channel dimension (see index where first upblock concat last downblock,...)
            y = up_block(y)

        y = y + x  # final output is added with original input x

        return y
