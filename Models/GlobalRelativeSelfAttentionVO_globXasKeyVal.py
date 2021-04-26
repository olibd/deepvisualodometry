from Models.GlobalRelativeSelfAttentionVO import GlobalRelativeSelfAttentionVO


class GlobalRelativeSelfAttentionVO_globXasKeyVal(GlobalRelativeSelfAttentionVO):
    def __init__(self, imsize1, imsize2, batchNorm=True, rnn_hidden_size=1000, rnn_dropout_out=0.5,
                 rnn_dropout_between=0, attention_dropout=0.65, num_attention_heads=8):
        """
        Uses x_rel as the Query and x_glob as the Key and Value in the forward method.
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

        x_rel, _ = self.attention_module2(x_rel, x_glob, x_glob)
        x_rel = x_rel.permute(1, 0, 2)
        x_rel = self.regulate2(x_rel)
        x_rel = x_rel.permute(1, 0, 2)
        x_rel, _ = self.attention_module3(x_rel, x_rel, x_rel)
        x_rel = x_rel.permute(1, 0, 2)
        x_rel = self.relativeFullyConnected(x_rel)

        return x_rel, x_glob_out