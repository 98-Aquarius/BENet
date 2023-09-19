from torch.nn import functional as F
from torch.nn.modules.activation import LeakyReLU
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention


class SR(nn.Module):
    def __init__(self, N, d_model=512):
        super(SR, self).__init__()
        # self.MLP = nn.Sequential(
        #     nn.Linear(N*d_model, N*d_model),
        #     nn.LeakyReLU(),
        #     nn.Linear(N*d_model, d_model),
        #     nn.LeakyReLU()
        # )
        self.lin = nn.Linear(N*d_model, d_model)
    def forward(self, x, layers, attention_mask = None, attention_weights = None):
        out = x
        outs = []
        for l in layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out)
        # outs = self.MLP(torch.cat(outs, -1))
        # out = 0.2 * outs + out
        cat_out = torch.cat(outs, -1)
        # gated = self.gated(cat_out)
        out = self.lin(cat_out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(Block, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)
        self.mlp = Mlp(in_features=d_model, hidden_features=4*d_model, drop=dropout)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        queries, keys, values = self.lnorm(queries), self.lnorm(keys), self.lnorm(values)
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = att + self.dropout(att)

        att = self.lnorm(att)
        att = self.mlp(att)
        att = att + self.dropout(att)
        return att

class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        # self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
        #                                           identity_map_reordering=identity_map_reordering,
        #                                           attention_module=attention_module, # ScaledDotProductAttention
        #                                           attention_module_kwargs=attention_module_kwargs) # {'m': args.m}
        #                              for _ in range(N)])
        self.layers = nn.ModuleList([Block(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module, # ScaledDotProductAttention
                                                  attention_module_kwargs=attention_module_kwargs) # {'m': args.m}
                                     for _ in range(N)])
        self.SR = SR(N, d_model)
        self.padding_idx = padding_idx

    def forward(self, input, attention_weights=None):

        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        out = self.SR(input, self.layers, attention_mask, attention_weights)

        return out, attention_mask


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, num_patches=49, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_in))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, input, attention_weights=None):
        mask = (torch.sum(input, dim=-1) == 0).unsqueeze(-1)
        input = input + self.pos_embed

        # out = F.relu(self.fc(input))
        out = self.dropout(self.fc(input))
        # out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)
        return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights)
