from torch.nn import functional as F
from torch.nn.modules.activation import LeakyReLU
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention

class SR(nn.Module):
    def __init__(self, N, d_model=512, seq_len = 49):
        super(SR, self).__init__()
        self.LG = nn.Sequential(
            nn.Linear(N*d_model, d_model),
        )
    def forward(self, x, layers, attention_mask = None, attention_weights = None):
        out = x
        outs = []
        for l in layers:
            out = l(out, out, out, attention_mask, attention_weights)
            outs.append(out)
        cat = torch.cat(outs, -1)
        out = self.LG(cat)
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
        self.gated = torch.nn.Sigmoid()

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
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
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.pos = nn.Parameter(torch.zeros(1, 49, d_in))

    def forward(self, input, attention_weights=None):
        mask = (torch.sum(input, dim=-1) == 0).unsqueeze(-1)
        out = F.relu(self.fc(input + self.pos))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)
        return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights)
