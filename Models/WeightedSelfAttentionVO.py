from torch import nn

from Models.SelfAttentionVO import SelfAttentionVO


class WeightedSelfAttentionVO(SelfAttentionVO):
    def __init__(self, imsize1, imsize2, batchNorm=True, rnn_hidden_size=1000, rnn_dropout_out=0.5,
                 rnn_dropout_between=0, attention_dropout=0.65, num_attention_heads=8):
        super().__init__(imsize1, imsize2, batchNorm, rnn_hidden_size, rnn_dropout_out, rnn_dropout_between,
                         attention_dropout, num_attention_heads)

        self.K1_weights = nn.Sequential(
            nn.Linear(in_features=2000, out_features=2000, bias=False),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True))

        self.Q1_weights = nn.Sequential(
            nn.Linear(in_features=2000, out_features=2000, bias=False),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True))

        self.K2_weights = nn.Sequential(
            nn.Linear(in_features=2000, out_features=2000, bias=False),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True))

        self.Q2_weights = nn.Sequential(
            nn.Linear(in_features=2000, out_features=2000, bias=False),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True))

        self.K3_weights = nn.Sequential(
            nn.Linear(in_features=2000, out_features=2000, bias=False),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True))

        self.Q3_weights = nn.Sequential(
            nn.Linear(in_features=2000, out_features=2000, bias=False),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = self._encode(x)
        x, hidden_state = self.rnn(x)
        K1 = self.K1_weights(x)
        Q1 = self.Q1_weights(x)
        x = x.permute(1, 0, 2)  # reshape to match attention module's expected input shape
        K1 = K1.permute(1, 0, 2)
        Q1 = Q1.permute(1, 0, 2)
        x, attention_output_weights = self.attention_module(Q1, K1, x)
        x = x.permute(1, 0, 2)
        x = self.regulate1(x)
        K2 = self.K2_weights(x)
        Q2 = self.Q2_weights(x)
        x = x.permute(1, 0, 2)  # reshape to match attention module's expected input shape
        K2 = K2.permute(1, 0, 2)
        Q2 = Q2.permute(1, 0, 2)
        x, attention_output_weights2 = self.attention_module2(Q2, K2, x)
        x = x.permute(1, 0, 2)
        x = self.regulate2(x)
        K3 = self.K3_weights(x)
        Q3 = self.Q3_weights(x)
        x = x.permute(1, 0, 2)  # reshape to match attention module's expected input shape
        K3 = K3.permute(1, 0, 2)
        Q3 = Q3.permute(1, 0, 2)
        x, attention_output_weights3 = self.attention_module3(Q3, K3, x)
        x = x.permute(1, 0, 2)  # reshape to match linear module's expected input shape
        x_rel = self.relativeFullyConnected(x)

        return x_rel
