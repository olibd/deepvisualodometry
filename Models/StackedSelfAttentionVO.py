import math

import torch
from torch import nn

from Models.SelfAttentionVO import SelfAttentionVO


class StackedSelfAttentionVO(SelfAttentionVO):
    """
    Stacked Multi attention head model using the Pytorch TransformerEncoder of the transformer network presented in
    "Attention is all you need" by Vaswani et Al.
    """
    def __init__(self, imsize1, imsize2, batchNorm=True,
                 rnn_hidden_size=1000,
                 rnn_dropout_out=0.5,
                 rnn_dropout_between=0,
                 attention_dropout=0.5,
                 num_attention_heads=8,
                 num_encoder_layer=3):
        self.num_encoder_layer = num_encoder_layer
        super().__init__(imsize1, imsize2, batchNorm, rnn_hidden_size, rnn_dropout_out,
                         rnn_dropout_between, attention_dropout=attention_dropout,
                         num_attention_heads=num_attention_heads)
        self.positional_encoder = self._get_positional_encoder()
        self.attention_module = self._get_positional_encoder()

    def forward(self, x):
        x = self._encode(x)
        x, hidden_state = self.rnn(x)
        x = x.permute(1, 0, 2)  # reshape to match attention module's expected input shape
        x = self.positional_encoder(x)
        x = self.attention_module(x)
        x = x.permute(1, 0, 2)  # reshape to match linear module's expected input shape
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

    def _get_positional_encoder(self):
        return StackedSelfAttentionVO.PrivatePositionalEncoding(self.rnn_hidden_size *2)

    class PrivatePositionalEncoding(nn.Module):
        """
        Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        """

        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super().__init__()
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