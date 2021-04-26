import math
from typing import Optional

import numpy
import torch
from torch import nn
from torch.autograd import Variable


class GlobalRelativeTransformerVO(nn.Module):

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
        self.global_encoder = self._get_transformer_encoder(n_layers=1)
        self.relative_decoder = self._get_transformer_decoder(n_layers=1)
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
        x_glob = self.global_encoder(x_glob)
        x_glob_out = x_glob.permute(1, 0, 2)
        x_glob_out = self.GlobalFullyConnected(x_glob_out)

        x_rel = x.permute(1, 0, 2)  # reshape to match attention module's expected input shape
        x_rel = self.relative_decoder(x_rel, x_glob)
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

    def _get_transformer_encoder(self, n_layers: int):
        layer = nn.TransformerEncoderLayer(d_model=self.rnn_hidden_size * 2, nhead=self.num_attention_heads,
                                           dropout=self.attention_dropout, dim_feedforward=2000)
        return nn.TransformerEncoder(layer, n_layers)

    def _get_transformer_decoder(self, n_layers: int):
        decoder_layer1 = GlobalRelativeTransformerVO._PrivateDecoderLayer(d_model=self.rnn_hidden_size * 2, nhead=self.num_attention_heads,
                                           dropout=self.attention_dropout, dim_feedforward=2000)
        return nn.TransformerDecoder(decoder_layer1, n_layers)

    def _tranformer_decoder(self, x, x_encoded):
        return self.decoder_layer1(x, x_encoded)

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

    def _get_positional_encoder(self, imsize1, imsize2):
        # Compute the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self._encode(__tmp)
        __tmp = self.rnn(__tmp)
        return GlobalRelativeTransformerVO.PrivatePositionalEncoding(int(numpy.prod(__tmp.size())))

    class PrivatePositionalEncoding(nn.Module):
        """
        Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        """

        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)

    class _PrivateRNNOutputDropout(nn.Module):
        def __init__(self, dropout_probability):
            super().__init__()
            self.dropout = nn.Dropout(dropout_probability)

        def forward(self, tuple: tuple):
            assert len(tuple) == 2, "tuple must be of size 2"
            prediction, hidden_state = tuple
            return self.dropout(prediction), hidden_state

    class _PrivateDecoderLayer(nn.TransformerDecoderLayer):
        def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                    memory_mask: Optional[torch.Tensor] = None,
                    tgt_key_padding_mask: Optional[torch.Tensor] = None,
                    memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """Pass the inputs (and mask) through the decoder layer. Inverts the Attention is all you need
            Query/Key-Value inputs: The output of the encoder (the memory parameter) will be used as Query
            and the output from the previous attention head will be used as the key-value pair.

            Args:
                tgt: the sequence to the decoder layer (required).
                memory: the sequence from the last layer of the encoder (required).
                tgt_mask: the mask for the tgt sequence (optional).
                memory_mask: the mask for the memory sequence (optional).
                tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
                memory_key_padding_mask: the mask for the memory keys per batch (optional).

            Shape:
                see the docs in Transformer class.
            """
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            tgt2 = self.multihead_attn(memory, tgt, tgt, attn_mask=tgt_mask,
                                       key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            return tgt

