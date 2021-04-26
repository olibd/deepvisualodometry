import math

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SnailVO(nn.Module):

    def __init__(self, imsize1, imsize2, seq_len, batchNorm=True):
        """
        @param seq_len:
        @param batchNorm:
        """
        super().__init__()

        self._setup_flownet_convolutional_layers(batchNorm)

        self.snail_module = self._get_snail_module(imsize1, imsize2, 2000, seq_len)

        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=2000, out_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=256, out_features=6))

    def forward(self, x):
        x = self._encode(x)
        x = self.snail_module(x)
        x = self.linear(x)

        return x

    def _encode(self, x) -> torch.Tensor:
        """
        @param x: (batch_size, seq_len, channel, width, height)
        @return: Tensor of shape (batch_size, stacked_sequence_len, flattened_convoluted_channel_width_height)
        """

        # stack the images into image pairs
        x = torch.cat((x[:, :-1], x[:, 1:]), dim=2)

        batch_size = x.size(0)
        seq_len = x.size(1)

        # Combine first 2 dimensions to make all segments its own batch such that the shape is
        # (batch, channel, width, height)
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))

        x = self._flownet_convolutional_encoder(x)

        # reshape the tensor so that it has shape:
        # (batch_size, stacked_sequence_len, flattened_convoluted_channel_width_height)
        x = x.view(batch_size, seq_len, -1)
        return x

    def _get_snail_module(self, imsize1, imsize2, output_size, sequence_length):
        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self._flownet_convolutional_encoder(__tmp)
        input_size = int(numpy.prod(__tmp.size()))

        return SnailModel(input_size, output_size, sequence_length)

    def _flownet_convolutional_encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv4(x)
        x = self.conv4_1(x)
        x = self.conv5(x)
        x = self.conv5_1(x)

        return self.conv6(x)

    def _setup_flownet_convolutional_layers(self, batchnorm):
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.conv1 = self._get_convolutional_layer(batchnorm, 6, 64, kernel_size=7, stride=2,
                                                   dropout=self.conv_dropout[0])
        self.conv2 = self._get_convolutional_layer(batchnorm, 64, 128, kernel_size=5, stride=2,
                                                   dropout=self.conv_dropout[1])
        self.conv3 = self._get_convolutional_layer(batchnorm, 128, 256, kernel_size=5, stride=2,
                                                   dropout=self.conv_dropout[2])
        self.conv3_1 = self._get_convolutional_layer(batchnorm, 256, 256, kernel_size=3, stride=1,
                                                     dropout=self.conv_dropout[3])
        self.conv4 = self._get_convolutional_layer(batchnorm, 256, 512, kernel_size=3, stride=2,
                                                   dropout=self.conv_dropout[4])
        self.conv4_1 = self._get_convolutional_layer(batchnorm, 512, 512, kernel_size=3, stride=1,
                                                     dropout=self.conv_dropout[5])
        self.conv5 = self._get_convolutional_layer(batchnorm, 512, 512, kernel_size=3, stride=2,
                                                   dropout=self.conv_dropout[6])
        self.conv5_1 = self._get_convolutional_layer(batchnorm, 512, 512, kernel_size=3, stride=1,
                                                     dropout=self.conv_dropout[7])
        self.conv6 = self._get_convolutional_layer(batchnorm, 512, 1024, kernel_size=3, stride=2,
                                                   dropout=self.conv_dropout[8])

    def _get_convolutional_layer(self, batchnorm, in_planes, out_planes, kernel_size, stride, dropout):
        """
        modified from https://github.com/ChiWeiHsiao/DeepVO-pytorch
        @param batchnorm:
        @param in_planes:
        @param out_planes:
        @param kernel_size:
        @param stride:
        @param dropout:
        @return:
        """
        if batchnorm:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)
            )

##############################################################################################################
# All code below this line was inspired by: https://github.com/sagelywizard/snail/blob/master/models.py

class SnailModel(nn.Module):
    """
    Arguments:
        output_size (int): number of output nodes
        sequence_length (int): k-shot. i.e. number of examples
    """
    def __init__(self, input_size, output_size, sequence_length):
        super().__init__()
        layer_count = math.ceil(math.log(sequence_length)/math.log(2))
        self.mod0 = AttentionBlock(input_size, 384, 192)
        self.dropout0 = nn.Dropout(0.5)
        self.mod1 = TCBlock(input_size+192, sequence_length, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.mod2 = AttentionBlock(input_size+192+128*layer_count, 1536, 768)
        self.dropout2 = nn.Dropout(0.5)
        self.mod3 = TCBlock(input_size+192+128*layer_count+768, sequence_length, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.mod4 = AttentionBlock(input_size+192+2*128*layer_count+768, 3072, 1536)
        self.dropout4 = nn.Dropout(0.5)
        self.out_layer = nn.Conv1d(input_size+192+2*128*layer_count+768+1536, output_size, 1)

    def forward(self, minibatch):
        """
        @param minibatch: shape is (batch_size, seq_len, channels)
        @return:
        """
        minibatch = minibatch.permute(0, 2, 1)
        out = self.mod0(minibatch)
        out = self.dropout0(out)
        out = self.mod1(out)
        out = self.dropout1(out)
        out = self.mod2(out)
        out = self.dropout2(out)
        out = self.mod3(out)
        out = self.dropout3(out)
        out = self.mod4(out)
        out = self.dropout4(out)
        out = self.out_layer(out)
        return out.permute(0, 2, 1)


##############################################################################################################
# All code below this line was taken directly from: https://github.com/sagelywizard/snail/blob/master/snail.py

class CausalConv1d(nn.Module):
    """A 1D causal convolution layer.
    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions per step, and T is the number of steps.
    Output: (B, D_out, T), where B is the minibatch size, D_out is the number
        of dimensions in the output, and T is the number of steps.
    Arguments:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = dilation
        self.causal_conv = nn.Conv1d(
            in_channels,
            out_channels,
            2,
            padding = self.padding,
            dilation = dilation
        )

    def forward(self, minibatch):
        return self.causal_conv(minibatch)[:, :, :-self.padding]


class DenseBlock(nn.Module):
    """Two parallel 1D causal convolution layers w/tanh and sigmoid activations
    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions of the input, and T is the number of steps.
    Output: (B, D_in+F, T), where where `B` is the minibatch size, `D_in` is the
        number of dimensions of the input, `F` is the number of filters, and `T`
        is the length of the input sequence.
    Arguments:
        in_channels (int): number of input channels
        filters (int): number of filters per channel
    """
    def __init__(self, in_channels, filters, dilation=1):
        super(DenseBlock, self).__init__()
        self.causal_conv1 = CausalConv1d(
            in_channels,
            filters,
            dilation=dilation
        )
        self.causal_conv2 = CausalConv1d(
            in_channels,
            filters,
            dilation=dilation
        )

    def forward(self, minibatch):
        tanh = F.tanh(self.causal_conv1(minibatch))
        sig = F.sigmoid(self.causal_conv2(minibatch))
        out = torch.cat([minibatch, tanh*sig], dim=1)
        return out


class TCBlock(nn.Module):
    """A stack of DenseBlocks which dilates to desired sequence length
    The TCBlock adds `ceil(log_2(seq_len))*filters` channels to the output.
    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions of the input, and T is the number of steps.
    Output: (B, D_in+F, T), where where `B` is the minibatch size, `D_in` is the
        number of dimensions of the input, `F` is the number of filters, and `T`
        is the length of the input sequence.
    Arguments:
        in_channels (int): channels for the input
        seq_len (int): length of the sequence. The number of denseblock layers
            is log base 2 of `seq_len`.
        filters (int): number of filters per channel
    """
    def __init__(self, in_channels, seq_len, filters):
        super(TCBlock, self).__init__()
        layer_count = math.ceil(math.log(seq_len)/math.log(2))
        blocks = []
        channel_count = in_channels
        for layer in range(layer_count):
            block = DenseBlock(channel_count, filters, dilation=2**layer)
            blocks.append(block)
            channel_count += filters
        self.blocks = nn.Sequential(*blocks)

    def forward(self, minibatch):
        return self.blocks(minibatch)


class AttentionBlock(nn.Module):
    """An attention mechanism similar to Vaswani et al (2017)
    The input of the AttentionBlock is `BxDxT` where `B` is the input
    minibatch size, `D` is the dimensions of each feature, `T` is the length of
    the sequence.
    The output of the AttentionBlock is `Bx(D+V)xT` where `V` is the size of the
    attention values.
    Arguments:
        input_dims (int): the number of dimensions (or channels) of each element
            in the input sequence
        k_size (int): the size of the attention keys
        v_size (int): the size of the attention values
    """
    def __init__(self, input_dims, k_size, v_size):
        super(AttentionBlock, self).__init__()
        self.key_layer = nn.Linear(input_dims, k_size)
        self.query_layer = nn.Linear(input_dims, k_size)
        self.value_layer = nn.Linear(input_dims, v_size)
        self.sqrt_k = math.sqrt(k_size)

    def forward(self, minibatch):
        minibatch = minibatch.permute(0,2,1)
        keys = self.key_layer(minibatch)
        queries = self.query_layer(minibatch)
        values = self.value_layer(minibatch)
        logits = torch.bmm(queries, keys.transpose(2,1))
        mask = logits.data.new(logits.size(1), logits.size(2)).fill_(1).bool()
        mask = torch.triu(mask, 1)
        mask = mask.unsqueeze(0).expand_as(logits)
        logits.data.masked_fill_(mask, float('-inf'))
        probs = F.softmax(logits / self.sqrt_k, dim=2)
        read = torch.bmm(probs, values)
        return torch.cat([minibatch, read], dim=2).permute(0,2,1)