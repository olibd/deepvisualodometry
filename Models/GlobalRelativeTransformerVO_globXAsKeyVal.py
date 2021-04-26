from torch import nn

from Models.GlobalRelativeTransformerVO import GlobalRelativeTransformerVO


class GlobalRelativeTransformerVO_globXAsKeyVal(GlobalRelativeTransformerVO):
    def __init__(self, imsize1, imsize2, batchNorm=True, rnn_hidden_size=1000, rnn_dropout_out=0.5,
                 rnn_dropout_between=0, attention_dropout=0.65, num_attention_heads=8):
        super().__init__(imsize1, imsize2, batchNorm, rnn_hidden_size, rnn_dropout_out, rnn_dropout_between,
                         attention_dropout, num_attention_heads)

    def _get_transformer_decoder(self, n_layers: int):
        decoder_layer1 = nn.TransformerDecoderLayer(d_model=self.rnn_hidden_size * 2, nhead=self.num_attention_heads,
                                           dropout=self.attention_dropout, dim_feedforward=2000)
        return nn.TransformerDecoder(decoder_layer1, n_layers)