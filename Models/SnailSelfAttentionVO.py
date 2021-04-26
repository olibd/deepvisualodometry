import math

import torch
from torch import nn

from Models.SelfAttentionVO import SelfAttentionVO


class SnailSelfAttentionVO(SelfAttentionVO):
    """
    Model using the SNAIL attention block presented in
    "A Simple Neural Attentive Meta-Learner" by Mishra et Al.
    """
    def __init__(self, imsize1, imsize2, batchNorm=True,
                 rnn_hidden_size=1000,
                 rnn_dropout_out=0.5,
                 rnn_dropout_between=0):
        super().__init__(imsize1, imsize2, batchNorm, rnn_hidden_size, rnn_dropout_out,
                         rnn_dropout_between, attention_dropout=0, num_attention_heads=0)

        self.linear = nn.Sequential(nn.Linear(in_features=self.rnn_hidden_size*4, out_features=256), # multiply by 2 for bidirectional
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Linear(in_features=256, out_features=6))

    def forward(self, x):
        x = self._encode(x)
        x, hidden_state = self.rnn(x)
        x = x.permute(0, 2, 1)
        x = self.attention_module(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)

        return x

    def _get_attention_module(self):
        return SnailSelfAttentionVO._AttentionBlock(self.rnn_hidden_size * 2, self.rnn_hidden_size * 2,
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
            super(SnailSelfAttentionVO._AttentionBlock, self).__init__()
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
            mask = logits.data.new(logits.size(1), logits.size(2)).fill_(1).bool()
            mask = torch.triu(mask, 1)
            mask = mask.unsqueeze(0).expand_as(logits)
            logits.data.masked_fill_(mask, float('-inf'))
            probs = torch.nn.functional.softmax(logits / self.sqrt_k, dim=2)
            read = torch.bmm(probs, values)
            return torch.cat([minibatch, read], dim=2).permute(0, 2, 1)