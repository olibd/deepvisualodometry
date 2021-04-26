from torch.nn import LayerNorm

from Models.SelfAttentionVO import SelfAttentionVO


class SkippedSelfAttention(SelfAttentionVO):
    def __init__(self, imsize1, imsize2, batchNorm=True, rnn_hidden_size=1000, rnn_dropout_out=0.5,
                 rnn_dropout_between=0, attention_dropout=0.65, num_attention_heads=8):
        super().__init__(imsize1, imsize2, batchNorm, rnn_hidden_size, rnn_dropout_out, rnn_dropout_between,
                         attention_dropout, num_attention_heads)
        self.layer_norm1 = LayerNorm(self.rnn_hidden_size * 2)
        self.layer_norm2 = LayerNorm(self.rnn_hidden_size * 2)
        self.layer_norm3 = LayerNorm(self.rnn_hidden_size * 2)

    def forward(self, x):
        x = self._encode(x)
        x, hidden_state = self.rnn(x)
        x = x.permute(1, 0, 2)  # reshape to match attention module's expected input shape
        x_att, attention_output_weights = self.attention_module(x, x, x)
        x = x + x_att
        x = self.layer_norm1(x)
        x = x.permute(1, 0, 2)
        x = self.regulate1(x)
        x = x.permute(1, 0, 2)
        x_att, attention_output_weights2 = self.attention_module2(x, x, x)
        x = x + x_att
        x = self.layer_norm2(x)
        x = x.permute(1, 0, 2)
        x = self.regulate2(x)
        x = x.permute(1, 0, 2)
        x_att, attention_output_weights3 = self.attention_module3(x, x, x)
        x = x + x_att
        x = self.layer_norm3(x)
        x = x.permute(1, 0, 2)  # reshape to match linear module's expected input shape
        x_rel = self.relativeFullyConnected(x)

        return x_rel