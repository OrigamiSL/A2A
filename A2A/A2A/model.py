import torch.nn as nn
from A2A.Layers import ELAN_block
from SecondStage.Embed import DataEmbedding
from SecondStage.MLP import FC_block


class A2A(nn.Module):
    def __init__(self, label_len, pred_list, enc_in, d_model, repr_dim, act, dropout,
                 kernel, attn_nums, pyramid):
        super().__init__()
        self.d_model = d_model
        self.label_len = label_len
        self.pred_list = pred_list
        self.enc_in = enc_in
        self.attn_nums = attn_nums
        self.pyramid = pyramid
        self.repr_dim = repr_dim

        self.enc_bed = [DataEmbedding(1, d_model, dropout) for i in range(pyramid)]
        self.enc_bed = nn.ModuleList(self.enc_bed)

        Pr_blocks = [ELAN_block(d_model, kernel, dropout, attn_nums - i) for i in range(pyramid)]
        self.Pr_blocks = nn.ModuleList(Pr_blocks)

        self.flatten = nn.Flatten()
        repr_layer = []

        for i in range(pyramid):
            in_channels = d_model * label_len // (2 ** i) // (attn_nums - i + 1)
            repr_layer.append(nn.Linear(in_channels, repr_dim))
        self.repr_layer = nn.ModuleList(repr_layer)

        MLP_list = [FC_block(repr_dim, pred_list[i], act=act, num=pyramid, dropout=dropout) for i in range(len(pred_list))]
        self.MLP = nn.ModuleList(MLP_list)

    def forward(self, x_enc, flag='first stage', index=0):
        B, L, D = x_enc.shape
        x_enc = x_enc.transpose(1, 2).contiguous().view(B * D, L).unsqueeze(-1)
        enc_output = []
        py = 0
        for embed, Pr_b in zip(self.enc_bed, self.Pr_blocks):
            embed_enc = embed(x_enc[:, -self.label_len // (2 ** py):, :])
            block_out = Pr_b(embed_enc)
            block_out = self.repr_layer[py](block_out.contiguous().view(B, D, -1))
            enc_output.append(block_out)
            py += 1
        if flag == 'first stage':
            return enc_output
        else:
            temp_out = self.MLP[index](enc_output)
            temp_out = temp_out.contiguous().view(B, D, self.pred_list[index])
            enc_out = temp_out.transpose(1, 2)
        return enc_out
