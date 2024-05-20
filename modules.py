import torch
import torch.nn.functional as F

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear', batch_norm=False):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)
        self.norm = torch.nn.BatchNorm1D(out_channels) if batch_norm else None

        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        if self.norm is None:
            return self.conv(signal)
        else:
            return self.norm(self.conv(signal))


class ConvReLUNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0, sepconv=False):
        super(ConvReLUNorm, self).__init__()
        if sepconv:
            self.conv_fn = SeparableConv
        else:
            self.conv_fn = torch.nn.Conv1d
        self.conv = self.conv_fn(in_channels, out_channels,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=(kernel_size // 2))
        self.norm = torch.nn.LayerNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal):
        out = F.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2).to(signal.dtype)
        return self.dropout(out)


class SeparableConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SeparableConv, self).__init__()
        self.depthwise = torch.nn.Conv1d(in_channels, in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv1d(in_channels, out_channels,
                                         kernel_size=1)

    def forward(self, signal):
        out = self.depthwise(signal)
        out = self.pointwise(out)
        return out