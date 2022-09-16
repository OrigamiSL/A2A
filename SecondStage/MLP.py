import torch
import torch.nn as nn


class FC_block(nn.Module):
    def __init__(self, in_channels, out_channels, act, num, dropout):
        super(FC_block, self).__init__()
        FC = [nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Dropout(dropout),
            nn.Tanh() if act == 'Tanh' else nn.GELU(),
            nn.Linear(in_channels, out_channels)) for _ in range(num)]
        self.FC = nn.ModuleList(FC)

    def forward(self, x_list):
        i = 0
        x_out = 0
        for x, fc in zip(x_list, self.FC):
            x_out += fc(x.clone().detach())
            i += 1
        return x_out / i
