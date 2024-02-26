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
        # self.gcn = GcnNet(hidden_size)
        # self.gat = GraghAttention(hidden_size)
        self.cat_connect = nn.Linear(2 * TEXT_DIM, TEXT_DIM)
        # self.weight = nn.Linear(hidden_size, hidden_size)
        # self.null = None
        # self.null1 = None

    # def GCN(self, adj, x):
    #     list1 = []
    #     B, _, _ = adj.shape
    #     b, _, _ = x.shape
    #     self.null = torch.zeros(50, 768)
    #     for i, j in zip(range(B), range(b)):
    #         shift_gcn = self.gcn(adj[i], x[j]) + self.null.to(device)
    #         self.null = shift_gcn
    #         list1.append(shift_gcn.unsqueeze(0))
    #     connect_shift_gcn = torch.cat(list1, dim=0)
    #
    #     return connect_shift_gcn
    #
    # def GAT(self, adj_a, x_a):
    #     list2 = []
    #     A, _, _ = adj_a.shape
    #     a, _, _ = x_a.shape
    #     self.null1 = torch.zeros(50, 768)
    #     for m, n in zip(range(A), range(a)):
    #         shift_gat = self.gat(adj_a[m], x_a[n]) + self.null1.to(device)
    #         self.null1 = shift_gat
    #         list2.append(shift_gat.unsqueeze(0))
    #     connect_shift_gat = torch.cat(list2, dim=0)
    #
    #     return connect_shift_gat

    def forward(self, text_embedding, visual=None, acoustic=None, visual_ids=None, acoustic_ids=None):
        # 视觉图卷积表示学习
        visual_ = self.visual_embedding(visual_ids)
        vis = visual_
        # shift_gcn_vv = self.GCN(visual_, visual_)
        # shift_gcn_vv = self.GAT(visual_, visual_) + visual_

        # 声学图卷积表示学习
        acoustic_ = self.acoustic_embedding(acoustic_ids)
        aco = acoustic_
        # shift_gcn_aa = self.GCN(acoustic_, acoustic_)
        # shift_gcn_aa = self.GAT(acoustic_, acoustic_) + acoustic_

        # 文本图卷积表示学习
        # shift_gcn_tt = self.GCN(text_embedding, text_embedding)
        # shift_gcn_tt = self.GAT(shift_gcn_tt, shift_gcn_tt) + self.GAT(shift_gcn_tt, shift_gcn_vv) + self.GAT(shift_gcn_tt, shift_gcn_aa) + text_embedding

        # 跨模态交互
        # 视觉嵌入文本
        visual_ = self.hv(text_embedding, visual_)
        # visual_ = self.ffn(visual_)

        # 声学嵌入文本
        acoustic_ = self.ha(text_embedding, acoustic_)
        # acoustic_ = self.ffn(acoustic_)

        # 基于文本的声视融合
        visual_acoustic = torch.cat((visual_, acoustic_), dim=-1)
        F1 = self.cat_connect(visual_acoustic)
        F1 = self.ffn(F1)
        F1 = F1.view(-1, 2)
        print(F1.shape)
        exit()
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
        # 残差连接
        embedding_shift = text_embedding + F1 + F2 + F3

        return embedding_shift

# # 图注意力
# class GraphAttentionLayer(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(GraphAttentionLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#
#         # Learnable weight matrices for linear transformations
#         self.W = nn.Parameter(torch.randn(50, out_features))
#         self.a = nn.Parameter(torch.randn(in_features, out_features))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         init.kaiming_uniform_(self.W)
#         init.kaiming_uniform_(self.a)
#
#     def forward(self, x, adj_matrix):
#         # Linear transformation
#         h = torch.mm(x.T, self.W)
#
#         # Attention mechanism
#         attention = torch.mm(torch.tanh(h), self.a)
#         attention = F.softmax(attention, dim=1)
#
#         # Weighted sum of neighbors' features
#         neighbors_features = torch.mm(adj_matrix, attention)
#
#         # Final output
#         output = torch.relu(neighbors_features)
#         return output
#
# class GraghAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(GraghAttention, self).__init__()
#         self.layer1 = GraphAttentionLayer(hidden_size, hidden_size)
#         self.layer2 = GraphAttentionLayer(hidden_size, hidden_size)
#
#     def forward(self, x, adj_matrix):
#         x = self.layer1(x, adj_matrix)
#         x = self.layer2(x, adj_matrix)
#         return x
#
# # 图卷积实现
# class GraphConvolution(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(GraphConvolution, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         init.kaiming_uniform_(self.weight)
#
#     def forward(self, adjacency, input_feature):
#         # 图卷积公式
#         support = torch.mm(input_feature, self.weight)
#         output = torch.sparse.mm(adjacency.T, support)
#
#         return output
#
# # 图卷积封装
# class GcnNet(nn.Module):
#     def __init__(self, hidden_size):
#         super(GcnNet, self).__init__()
#         self.gcn1 = GraphConvolution(hidden_size, hidden_size)
#         self.gcn2 = GraphConvolution(hidden_size, hidden_size)
#
#     def forward(self, adjacency, feature):
#         h = F.relu(self.gcn1(adjacency, feature))
#         logits = self.gcn2(adjacency.permute(1, 0), h)
#
#         return logits
#
# # 位置编码
# class PositionalEncoding(nn.Module):
#     "Implement the PE function."
#
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#
#         if d_model % 2 != 0:
#             raise ValueError("ERROR!".format(d_model))
#
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
#                              -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position.float() * div_term)
#         pe[:, 1::2] = torch.cos(position.float() * div_term)
#         pe = pe.unsqueeze(1)
#         self.register_buffer('pe', pe)
#         self.d_model = d_model
#         self.dropout = nn.Dropout(p=0.1)
#
#     def forward(self, emb, step=None):
#
#         emb = emb + self.pe[:emb.size(0)]
#         emb = self.dropout(emb)
#         return emb

# 多头注意力
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, head_num=1):
        super(SelfAttention, self).__init__()
        self.head_num = head_num
        self.s_d = hidden_size // self.head_num
        self.all_head_size = self.head_num * self.s_d
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
        # self.position = PositionalEncoding(TEXT_DIM)

    def transpose_for_scores(self, x):
        x = x.view(x.size(0), x.size(1), self.head_num, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, text_embedding, embedding):
        Q = self.Wq(text_embedding)
        # 位置编码
        # Q = self.position(Q)
        K = self.Wk(embedding)
        # 位置编码
        # K = self.position(K)
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





