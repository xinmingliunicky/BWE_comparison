import torch.nn as nn
from utils import conv1d_same


# write a class for ResBlock
# conv refers to the convolution type (conv1d_same / conv1d_halve) to be used
class ResBlock(nn.Module):
    def __init__(self, conv, n_hidden_ch, kernel_size,
                 batchnorm=False, dropout=0.0,
                 activation=nn.ReLU(True), res_scale=0.1):

        super(ResBlock, self).__init__()

        bias = not batchnorm  # if use batchnorm, no bias presents; vice versa

        layers = []
        for i in range(2):
            layers.append(conv(n_hidden_ch, n_hidden_ch, kernel_size, bias=bias))
            if batchnorm:
                layers.append(nn.BatchNorm1d(n_hidden_ch))
            if dropout:
                layers.append(nn.Dropout(dropout))
            if i == 0:
                layers.append(activation)  # so that activation layer is added to the middle of 2 convs

        self.body = nn.Sequential(*layers)
        self.res_scale = res_scale


    # now we have a model self.body
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)  # scale of residual
        res += x  # add with original input
        return res




# now the whole ResNet
class Model(nn.Module):
    # default no dropout, no batchnorm, always use padding=0
    def __init__(self, batchnorm=False, dropout=0.0):

        super(Model, self).__init__()

        assert not (batchnorm and dropout)

        self.batchnorm = batchnorm
        self.bias = not self.batchnorm  # if use batchnorm, no bias presents; vice versa

        self.n_res_blocks = 15  # number of ResBlock
        # Mono/Stereo (Input layer & Output layer channel size)
        self.n_input_ch = 2
        self.n_output_ch = 2
        self.n_hidden_ch = 512  # channel size
        self.kernel_size = 7  # kernel size
        self.activation = nn.ReLU(inplace=True)  # activation layer
        self.res_scaling = 0.1  # scale of residual

        head = [conv1d_same(self.n_input_ch, self.n_hidden_ch, self.kernel_size)]  # the first layer

        # duplicate 15 ResBlocks
        body = [ResBlock(conv1d_same, self.n_hidden_ch, self.kernel_size, activation=self.activation, 
                         res_scale=self.res_scaling, batchnorm=self.batchnorm, dropout=dropout)
                for _ in range(self.n_res_blocks)]

        tail = [conv1d_same(self.n_hidden_ch, self.n_output_ch, 1)]  # the last layer

        self.model = nn.Sequential(*head, *body, *tail)


    def forward(self, x):
        input_ = x  # input x
        x = self.model(x)  # pipe through the model
        x += input_  # final output is added with input
        return x


