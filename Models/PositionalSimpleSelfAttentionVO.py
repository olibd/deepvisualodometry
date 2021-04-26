import math

import torch

from Models.SimpleSelfAttentionVO import SimpleSelfAttentionVO


class PositionalSimpleSelfAttentionVO(SimpleSelfAttentionVO):

    def __init__(self, imsize1, imsize2, batchNorm=True, rnn_hidden_size=1000, rnn_dropout_out=0.5,
                 rnn_dropout_between=0, attention_dropout=0.65, num_attention_heads=8):
        super().__init__(imsize1, imsize2, batchNorm, rnn_hidden_size, rnn_dropout_out,
                 rnn_dropout_between, attention_dropout, num_attention_heads)
        self.positional_encoder = self._get_positional_encoder()

    def forward(self, x):
        x = self._encode(x)
        x = self.linear_vectorizer(x)
        x = x.permute(1, 0, 2)  # reshape to match attention module's expected input shape
        x = self.positional_encoder(x)
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

    def _get_positional_encoder(self):
        return PositionalSimpleSelfAttentionVO.PrivatePositionalEncoding(self.rnn_hidden_size * 2)

    class PrivatePositionalEncoding(torch.nn.Module):
        """
        Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        """

        def __init__(self, d_model, dropout=0.1, max_len=2000):
            super(PositionalSimpleSelfAttentionVO.PrivatePositionalEncoding, self).__init__()
            self.dropout = torch.nn.Dropout(p=dropout)

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