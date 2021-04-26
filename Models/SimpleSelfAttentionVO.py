import numpy
import torch
from torch import nn
from torch.autograd import Variable
from Models.SelfAttentionVO import SelfAttentionVO


class SimpleSelfAttentionVO(SelfAttentionVO):


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
        super().__init__(imsize1, imsize2)
        self.attention_dropout = attention_dropout
        self.num_attention_heads = num_attention_heads
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_dropout_out = rnn_dropout_out
        self.rnn_dropout_between = rnn_dropout_between

        self._setup_flownet_convolutional_layers(batchNorm)
        self.linear_vectorizer = self._linear_vectorizer(imsize1, imsize2)
        self.attention_module = self._get_attention_module()
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

    def forward(self, x):
        x = self._encode(x)
        x = self.linear_vectorizer(x)
        x = x.permute(1, 0, 2)  # reshape to match attention module's expected input shape
        x, attention_output_weights = self.attention_module(x, x, x)
        x = x.permute(1, 0, 2)
        x = self.regulate1(x)
        x = x.permute(1, 0, 2)
        x, attention_output_weights2 = self.attention_module2(x, x, x)
        x = x.permute(1, 0, 2)
        x = self.regulate2(x)
        x = x.permute(1, 0, 2)
        x, attention_output_weights3 = self.attention_module3(x, x, x)
        x = x.permute(1, 0, 2)  # reshape to match linear module's expected input shape
        x_rel = self.relativeFullyConnected(x)

        return x_rel

    def _linear_vectorizer(self, imsize1, imsize2):
        # Compute the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self._flownet_convolutional_encoder(__tmp)
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=int(numpy.prod(__tmp.size())), out_features=self.rnn_hidden_size *2),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
        )
