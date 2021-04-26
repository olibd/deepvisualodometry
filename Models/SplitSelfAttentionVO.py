import torch
from torch import nn

from Models.SelfAttentionVO import SelfAttentionVO


class SplitSelfAttentionVO(SelfAttentionVO):
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

        super().__init__(imsize1, imsize2, batchNorm, rnn_hidden_size, rnn_dropout_out, rnn_dropout_between,
                         attention_dropout, num_attention_heads)
        self.relativeFullyConnected = nn.Sequential(
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=2000, out_features=256))
        self.relativeRotationFullyConnected = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=False),
            nn.Linear(in_features=256, out_features=3))
        self.relativeLocationFullyConnected = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=False),
            nn.Linear(in_features=256, out_features=3))

    def forward(self, x):
        x = self._encode(x)
        x, hidden_state = self.rnn(x)
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
        x_relRot = self.relativeRotationFullyConnected(x_rel)
        x_relLoc = self.relativeLocationFullyConnected(x_rel)

        return torch.cat((x_relRot, x_relLoc), 2)