import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def Simclr_pre(image, text):
    i_b, i_dim1, i_dim2 = image.shape

    pad_l = pad_sequence(text)
    t_b, t_dim1, t_dim2 = pad_l.shape


    flat_l = torch.flatten(pad_l, dim=1)
    flat_i = torch.flatten(image, dim=1)


    t_mlp = nn.Linear(t_dim1*t_dim2, 512)
    i_mlp = nn.Linear(i_dim1*i_dim2, 512)

    return t_mlp(flat_l), i_mlp(flat_i)



