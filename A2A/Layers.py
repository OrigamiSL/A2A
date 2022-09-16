import torch.nn as nn
import torch
from torch.nn.utils import weight_norm


class ConvLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, dropout=0, s=2, dilation=1):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.kernel = kernel
        self.downConv = weight_norm(nn.Conv1d(in_channels=c_in,
                                              out_channels=c_in,
                                              padding=(kernel - 1) * dilation // 2,
                                              stride=s,
                                              kernel_size=kernel,
                                              dilation=dilation))
        self.activation1 = nn.Tanh()
        self.actConv = weight_norm(nn.Conv1d(in_channels=c_in,
                                             out_channels=c_out,
                                             padding=1,
                                             stride=1,
                                             kernel_size=3))
        self.activation2 = nn.Tanh()
        self.sampleConv = nn.Conv1d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=1) if c_in != c_out else None
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if s != 1 else None

    def forward(self, x):
        y = x.clone()
        if self.sampleConv is not None:
            y = self.sampleConv(y.permute(0, 2, 1)).transpose(1, 2)
        if self.pool is not None:
            y = self.pool(y.permute(0, 2, 1)).transpose(1, 2)
        x = self.dropout(self.downConv(x.permute(0, 2, 1)))
        x = self.activation1(x).transpose(1, 2)
        x = self.dropout(self.actConv(x.permute(0, 2, 1)))
        x = self.activation2(x).transpose(1, 2)
        x = x + y
        return x


class ConvBlock(nn.Module):
    def __init__(self, c_in, kernel=3, dropout=0, layer=1, s=2):
        super(ConvBlock, self).__init__()
        self.s = s
        if s == 2:
            Conv = [nn.Sequential(ConvLayer(c_in, c_in, kernel, dropout, s=2),
                                  ConvLayer(c_in, c_in, kernel, dropout, s=1)) for i in range(layer)]
        else:
            Conv = [nn.Sequential(ConvLayer(c_in, c_in, kernel, dropout, s=2),
                                  ConvLayer(c_in, c_in, kernel, dropout, s=1)) for i in range(layer - 1)]
            SubConv1 = nn.Sequential(ConvLayer(c_in, c_in, kernel, dropout, s=2),
                                     ConvLayer(c_in, c_in, kernel, dropout, s=1))
            SubConv2 = nn.Sequential(ConvLayer(c_in, c_in, kernel, dropout, s=2),
                                     ConvLayer(c_in, c_in, kernel, dropout, s=1),
                                     ConvLayer(c_in, c_in, kernel, dropout, s=1),
                                     ConvLayer(c_in, c_in, kernel, dropout, s=1)
                                     )
            SubConv = [SubConv1, SubConv2]
            self.SubConv = nn.ModuleList(SubConv)
        self.conv = nn.ModuleList(Conv)

    def forward(self, x):
        for cons in self.conv:
            x = cons(x)
        if self.s == 1:
            out = []
            for SubConvBlock in self.SubConv:
                out.append(SubConvBlock(x))
            out = torch.cat(out, dim=1)
            return out
        return x


class ELAN_block(nn.Module):
    def __init__(self, d_model, kernel, dropout, attn_nums):
        super(ELAN_block, self).__init__()
        self.d_model = d_model
        self.attn_nums = attn_nums
        pro_conv = [ConvBlock(self.d_model // (attn_nums + 1), kernel=kernel, dropout=dropout, layer=i + 1)
                    for i in range(attn_nums - 1)]
        pro_conv.append(ConvBlock(self.d_model // (attn_nums + 1), kernel=kernel,
                                  dropout=dropout, layer=attn_nums, s=1))
        self.pro_conv = nn.ModuleList(pro_conv)

    def forward(self, x):
        out = []
        part_x = torch.split(x, self.d_model // (self.attn_nums + 1), dim=-1)
        for i, conv in enumerate(self.pro_conv):
            out.append(conv(part_x[i]))
        out = torch.cat(out, dim=1)
        return out
