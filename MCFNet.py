import torch
import torch.nn as nn
import torch.nn.functional as F
from config.global_configs import *
import numpy as np
import scipy.sparse as sp
from torch.nn import init
from torchsummary import summary
from thop import profile, clever_format

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CE(nn.Module):
    def __init__(self, ):
        super(CE, self).__init__()
        self.visual_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.acoustic_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.hv = SelfAttention(768)
        self.ha = SelfAttention(768)
        self.ht = SelfAttention(768)
        self.ffn = FFN()
        self.cat_connect = nn.Linear(2 * TEXT_DIM, TEXT_DIM)

    def forward(self, text_embedding, visual=None, acoustic=None, visual_ids=None, acoustic_ids=None):
        visual_ = self.visual_embedding(visual_ids)
        acoustic_ = self.acoustic_embedding(acoustic_ids)
        visual_ = self.hv(text_embedding, visual_)
        acoustic_ = self.ha(text_embedding, acoustic_)
        visual_acoustic = torch.cat((visual_, acoustic_), dim=-1)
        F1 = self.cat_connect(visual_acoustic)
        F1 = self.ffn(F1)
        F1 = F1.view(-1, 2)
        # NIRM
        v = self.hv(visual_, acoustic_)
        a = self.ha(acoustic_, visual_)
        v_a = torch.cat((v, a), dim=-1)
        F2 = self.cat_connect(v_a)
        F2 = self.ffn(F2)
        F2 = self.ffn(F2)
        # TIIM
        F3 = self.ht(text_embedding, F1)
        F3 = self.ffn(F3)
        F3 = self.ffn(F3)
        # Residual connection
        embedding_shift = text_embedding + F1 + F2 + F3

        return embedding_shift


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        if d_model % 2 != 0:
            raise ValueError("ERROR!".format(d_model))

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.d_model = d_model
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, emb, step=None):

        emb = emb + self.pe[:emb.size(0)]
        emb = self.dropout(emb)
        return emb

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, head_num=1):
        super(SelfAttention, self).__init__()
        self.head_num = head_num
        self.s_d = hidden_size // self.head_num
        self.all_head_size = self.head_num * self.s_d
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
        self.position = PositionalEncoding(TEXT_DIM)

    def transpose_for_scores(self, x):
        x = x.view(x.size(0), x.size(1), self.head_num, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, text_embedding, embedding):
        Q = self.Wq(text_embedding)
        Q = self.position(Q)
        K = self.Wk(embedding)
        K = self.position(K)
        V = self.Wv(embedding)
        Q = self.transpose_for_scores(Q)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)

        weight_score = torch.matmul(Q, K.transpose(-1, -2))
        weight_prob = nn.Softmax(dim=-1)(weight_score * 8)

        context_layer = torch.matmul(weight_prob, V)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


# 前馈神经网络
class FFN(nn.Module):
    def __init__(self, ):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.silu(x)
        x = self.linear2(x)

        return x





