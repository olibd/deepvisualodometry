from torch import nn

from Models.SelfAttentionVO import SelfAttentionVO


class SelfAttentionVO_GlobRelOutput(SelfAttentionVO):
    def __init__(self, imsize1, imsize2, batchNorm=True, rnn_hidden_size=1000, rnn_dropout_out=0.5,
                 rnn_dropout_between=0, attention_dropout=0.65, num_attention_heads=8):
        """
        Returns a tuple composed of relative pose estimation, global pose estimation in the forward method
        @param imsize1:
        @param imsize2:
        @param batchNorm:
        @param rnn_hidden_size:
        @param rnn_dropout_out:
        @param rnn_dropout_between:
        @param attention_dropout:
        @param num_attention_heads:
        """
        super().__init__(imsize1, imsize2, batchNorm, rnn_hidden_size, rnn_dropout_out, rnn_dropout_between,
                         attention_dropout, num_attention_heads)

        self.globalFullyConnected = nn.Sequential(
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=2000, out_features=256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=256, out_features=6))

    def forward(self, x) -> tuple:
        """

        @param x:
        @return: tuple composed of relative pose estimation, global pose estimation
        """
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
        x_glob = self.globalFullyConnected(x)

        return x_rel, x_glob