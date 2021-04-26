from torch import nn

from Models.Common import CoordConv2d
from Models.SelfAttentionVO import SelfAttentionVO


class CoordConvSelfAttentionVO(SelfAttentionVO):
    def _get_convolutional_layer(self, batchnorm, in_planes, out_planes, kernel_size, stride, dropout):
        """
        modified from https://github.com/ChiWeiHsiao/DeepVO-pytorch
        @param batchnorm:
        @param in_planes:
        @param out_planes:
        @param kernel_size:
        @param stride:
        @param dropout:
        @return:
        """
        if batchnorm:
            return nn.Sequential(
                CoordConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)
            )
        else:
            return nn.Sequential(
                CoordConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                          bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)
            )