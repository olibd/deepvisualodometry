import math

import numpy
import torch
from torch import nn
from torch.autograd import Variable


class OrdinarySelfAttentionVO(nn.Module):
    def __init__(self, imsize1, imsize2, batchNorm=True, rnn_hidden_size=1000, rnn_dropout_out=0.5,
                 rnn_dropout_between=0, attention_dropout=0.1, num_attention_heads=6):
        """

        @param imsize1:
        @param imsize2:
        @param batchNorm:
        @param rnn_hidden_size:
        @param rnn_dropout_out:
        @param rnn_dropout_between:
        @param attention_dropout: default is 0.1 based on the dropout used by nn.TransformerEncoderLayer
        """
        super().__init__()
        self.attention_dropout = attention_dropout
        self.num_attention_heads = num_attention_heads

        self.convolutional_encoder = self._get_convolutional_encoder(batchNorm)
        self.rnn = self._get_recurrent_layer(imsize1, imsize2)
        self.attention_module = self._get_attention_module()
        self.linear = nn.Sequential(nn.Linear(in_features=self.rnn_hidden_size*2, out_features=256), # multiply by 2 for bidirectional
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Linear(in_features=256, out_features=6))

    def forward(self, x):
        x = self._encode(x)
        x = self.rnn(x)
        x = self.attention_module(x, x, x)
        x = self.linear(x)

        return x

    def _encode(self, x):
        # x: (batch, seq_len, channel, width, height)
        # stack_image
        x = torch.cat((x[:, :-1], x[:, 1:]), dim=2)
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        x = self.convolutional_encoder(x)
        x = x.view(batch_size, seq_len, -1)
        return x

    def _get_attention_module(self):
        return nn.MultiheadAttention(embed_dim=self.rnn_hidden_size * 2, num_heads=self.num_attention_heads,
                                     dropout=self.attention_dropout)

    def _get_recurrent_layer(self, imsize1, imsize2):
        # Compute the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self.encode_image(__tmp)
        return nn.Sequential(
            nn.LSTM(
                input_size=int(numpy.prod(__tmp.size())),
                hidden_size=self.rnn_hidden_size,
                num_layers=2,
                dropout=self.rnn_dropout_between,
                batch_first=True,
                bidirectional=True
            ),
            nn.Dropout(self.rnn_dropout_out)
        )

    def _get_convolutional_encoder(self, batchnorm):
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.conv1 = self._get_convolutional_layer(batchnorm, 6, 64, kernel_size=7, stride=2, dropout=self.conv_dropout[0])
        self.conv2 = self._get_convolutional_layer(batchnorm, 64, 128, kernel_size=5, stride=2, dropout=self.conv_dropout[1])
        self.conv3 = self._get_convolutional_layer(batchnorm, 128, 256, kernel_size=5, stride=2, dropout=self.conv_dropout[2])
        self.conv3_1 = self._get_convolutional_layer(batchnorm, 256, 256, kernel_size=3, stride=1, dropout=self.conv_dropout[3])
        self.conv4 = self._get_convolutional_layer(batchnorm, 256, 512, kernel_size=3, stride=2, dropout=self.conv_dropout[4])
        self.conv4_1 = self._get_convolutional_layer(batchnorm, 512, 512, kernel_size=3, stride=1, dropout=self.conv_dropout[5])
        self.conv5 = self._get_convolutional_layer(batchnorm, 512, 512, kernel_size=3, stride=2, dropout=self.conv_dropout[6])
        self.conv5_1 = self._get_convolutional_layer(batchnorm, 512, 512, kernel_size=3, stride=1, dropout=self.conv_dropout[7])
        self.conv6 = self._get_convolutional_layer(batchnorm, 512, 1024, kernel_size=3, stride=2, dropout=self.conv_dropout[8])

        return nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv3_1,
            self.conv4,
            self.conv4_1,
            self.conv5,
            self.conv5_1,
            self.conv6
        )

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


class StackedOrdinarySelfAttentionVO(OrdinarySelfAttentionVO):
    def __init__(self, imsize1, imsize2, batchNorm=True,
                 rnn_hidden_size=1000,
                 rnn_dropout_out=0.5,
                 rnn_dropout_between=0,
                 attention_dropout=0.1,
                 num_attention_heads=6,
                 num_encoder_layer=6):
        self.num_encoder_layer = num_encoder_layer
        super().__init__(imsize1, imsize2, batchNorm, rnn_hidden_size, rnn_dropout_out,
                         rnn_dropout_between, attention_dropout=attention_dropout,
                         num_attention_heads=num_attention_heads)

    def forward(self, x):
        x = self._encode(x)
        x = self.rnn(x)
        x = self.attention_module(x)
        x = self.linear(x)

        return x

    def _get_attention_module(self):
        """
        Inspired by
        https://pytorch.org/docs/master/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder
        @param attention_dropout:
        @param num_attention_heads:
        @return:
        """
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.rnn_hidden_size * 2, nhead=self.num_attention_heads,
                                                   dropout=self.attention_dropout)
        return nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layer)


class SnailOrdinarySelfAttentionVO(OrdinarySelfAttentionVO):

    def __init__(self, imsize1, imsize2, batchNorm=True,
                 rnn_hidden_size=1000,
                 rnn_dropout_out=0.5,
                 rnn_dropout_between=0):
        super().__init__(imsize1, imsize2, batchNorm, rnn_hidden_size, rnn_dropout_out,
                         rnn_dropout_between, attention_dropout=0, num_attention_heads=0)

    def forward(self, x):
        x = self._encode(x)
        x = self.rnn(x)
        x = self.attention_module(x)
        x = self.linear(x)

        return x

    def _get_attention_module(self):
        return SnailOrdinarySelfAttentionVO._AttentionBlock(self.rnn_hidden_size * 2, self.rnn_hidden_size * 2,
                                                            self.rnn_hidden_size * 2)

    class _AttentionBlock(nn.Module):
        """
        Taken from: https://github.com/sagelywizard/snail/blob/master/snail.py

        An attention mechanism similar to Vaswani et al (2017)
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
            super(SnailOrdinarySelfAttentionVO._AttentionBlock, self).__init__()
            self.key_layer = nn.Linear(input_dims, k_size)
            self.query_layer = nn.Linear(input_dims, k_size)
            self.value_layer = nn.Linear(input_dims, v_size)
            self.sqrt_k = math.sqrt(k_size)

        def forward(self, minibatch):
            minibatch = minibatch.permute(0, 2, 1)
            keys = self.key_layer(minibatch)
            queries = self.query_layer(minibatch)
            values = self.value_layer(minibatch)
            logits = torch.bmm(queries, keys.transpose(2, 1))
            mask = logits.data.new(logits.size(1), logits.size(2)).fill_(1).byte()
            mask = torch.triu(mask, 1)
            mask = mask.unsqueeze(0).expand_as(logits)
            logits.data.masked_fill_(mask, float('-inf'))
            probs = torch.nn.functional.softmax(logits / self.sqrt_k, dim=2)
            read = torch.bmm(probs, values)
            return torch.cat([minibatch, read], dim=2).permute(0, 2, 1)