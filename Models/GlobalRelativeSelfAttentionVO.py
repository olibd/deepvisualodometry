import numpy
import torch
from torch import nn
from torch.autograd import Variable


class GlobalRelativeSelfAttentionVO(nn.Module):
    """
    Single Multi attention head model using the Pytorch Multiattention head which is similar to the one used in
    "Attention is all you need" by Vaswani et Al.
    """

    def __init__(self, imsize1, imsize2, batchNorm=True, rnn_hidden_size=1000, rnn_dropout_out=0.5,
                 rnn_dropout_between=0, attention_dropout=0.65, num_attention_heads=8):
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
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_dropout_out = rnn_dropout_out
        self.rnn_dropout_between = rnn_dropout_between

        self._setup_flownet_convolutional_layers(batchNorm)
        self.rnn = self._get_recurrent_layer(imsize1, imsize2)
        self.attention_module_relative = self._get_attention_module()
        self.attention_module_global = self._get_attention_module()
        self.regulate1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True))
        self.attention_module2 = self._get_attention_module()
        self.regulate2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True))
        self.attention_module3 = self._get_attention_module()
        self.relativeFullyConnected = nn.Sequential(
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=2000, out_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=256, out_features=6))
        self.GlobalFullyConnected = nn.Sequential(
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=2000, out_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=256, out_features=6))

    def forward(self, x):
        x = self._encode(x)
        x, hidden_state = self.rnn(x)

        x_glob = x.permute(1, 0, 2)  # reshape to match attention module's expected input shape
        x_glob, _ = self.attention_module_global(x_glob, x_glob, x_glob)
        x_glob_out = x_glob.permute(1, 0, 2)
        x_glob_out = self.GlobalFullyConnected(x_glob_out)

        x_rel = x.permute(1, 0, 2)  # reshape to match attention module's expected input shape
        x_rel, _ = self.attention_module_relative(x_rel, x_rel, x_rel)
        x_rel = x_rel.permute(1, 0, 2)
        x_rel = self.regulate1(x_rel)
        x_rel = x_rel.permute(1, 0, 2)

        x_rel, _ = self.attention_module2(x_glob, x_rel, x_rel)
        x_rel = x_rel.permute(1, 0, 2)
        x_rel = self.regulate2(x_rel)
        x_rel = x_rel.permute(1, 0, 2)
        x_rel, _ = self.attention_module3(x_rel, x_rel, x_rel)
        x_rel = x_rel.permute(1, 0, 2)
        x_rel = self.relativeFullyConnected(x_rel)

        return x_rel, x_glob_out

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

    def _get_attention_module(self):
        return nn.MultiheadAttention(embed_dim=self.rnn_hidden_size * 2, num_heads=self.num_attention_heads,
                                     dropout=self.attention_dropout)

    def _get_recurrent_layer(self, imsize1, imsize2):
        # Compute the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self._flownet_convolutional_encoder(__tmp)
        return nn.Sequential(
            nn.LSTM(
                input_size=int(numpy.prod(__tmp.size())),
                hidden_size=self.rnn_hidden_size,
                num_layers=2,
                dropout=self.rnn_dropout_between,
                batch_first=True,
                bidirectional=True
            ),
            self._PrivateRNNOutputDropout(self.rnn_dropout_out)
        )

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

    class _PrivateRNNOutputDropout(nn.Module):
        def __init__(self, dropout_probability):
            super().__init__()
            self.dropout = nn.Dropout(dropout_probability)

        def forward(self, tuple: tuple):
            assert len(tuple) == 2, "tuple must be of size 2"
            prediction, hidden_state = tuple
            return self.dropout(prediction), hidden_state
