from matplotlib import image
import torch
import torch.nn.functional as F
from torch import nn
import copy
from models.containers import ModuleList
from models.transformer.utils import sinusoid_encoding_table
from models.beam_search import *
from ..captioning_model import CaptioningModel
from torch.nn.parameter import Parameter
import math

class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        #self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres= shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        #self.reset_parameters()
        self.l1 = nn.Linear(fea_dim, mem_dim)
        self.l2 = nn.Linear(mem_dim, fea_dim)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        #print(input.shape)
        #att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = self.l1(input)
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            att_weight = F.normalize(att_weight, p=1, dim=1)
            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
        #mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        #output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        output = self.l2(att_weight)
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):


        y_and = self.memory(input)
        
        y = y_and['output']
        att = y_and['att']

        return {'output': y, 'att': att}

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder, num_clusters, vocab_size, max_len, padding_idx, text_d_model=512):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.text_d_model = text_d_model
        self.num_clusters=num_clusters
        self.padding_idx = padding_idx
        self.word_emb = nn.Embedding(vocab_size, text_d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, text_d_model, 0), freeze=True)

        self.memory = MemModule(100, 2048)

        self.softmax = nn.Softmax(dim=-1)
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, mode, images, seq=None, max_len=None, eos_idx=None, beam_size=None, out_size=1, return_probs=False):
        '''
        images: torch.Size([50, 49, 2048])
        seq: torch.Size([50, 27])
        '''
        if mode == 'xe':
            bs, _, vis_dim = images.size()
            # Grid feature
            grid_enc_output, grid_mask_enc = self.encoder(images)

            #Memory
            memory = self.memory(images)
            memory_feature = memory['output']
            #print(memory_feature.shape)
            memory_enc_output, memory_mask_enc = self.encoder(memory_feature)

            out, mask = torch.cat([grid_enc_output, memory_enc_output], dim=1), torch.cat([grid_mask_enc, memory_mask_enc], dim=-1)

            dec_output, encoder_out, img_out = self.decoder(seq, out, mask)

            return dec_output, encoder_out, img_out
            # return dec_output

        elif mode == 'rl':
            bs = BeamSearch(self, max_len, eos_idx, beam_size)
            return bs.apply(images, out_size, return_probs)

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                grid_enc_output, grid_mask_enc = self.encoder(visual)
                bs, _, vis_dim = visual.size()
                self.enc_output, self.mask_enc = grid_enc_output, grid_mask_enc

                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long() # self.bos_idx: '<bos>'
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output
        lanlan, _, _ = self.decoder(it, self.enc_output, self.mask_enc)
        return lanlan


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            #self.models[i].load_state_dict(state_dict_i)
            self.models[i].load_state_dict({k.replace('module.', ''):v for k,v in state_dict_i.items()})

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
