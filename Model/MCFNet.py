import torch.nn as nn
from config.global_configs import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MCFNet(nn.Module):
    def __init__(self, ):
        super(MCFNet, self).__init__()
        self.visual_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.acoustic_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size)
        self.hv = SelfAttention(768)
        self.ha = SelfAttention(768)
        self.ht = MultiheadAttention(768)
        self.ffn = FFN()
        self.cat_connect = nn.Linear(2 * TEXT_DIM, TEXT_DIM)
        self.norm = nn.Softmax(dim=-1)

    def forward(self, text_embedding, visual=None, acoustic=None, visual_ids=None, acoustic_ids=None):
        visual_ = self.visual_embedding(visual_ids)
        acoustic_ = self.acoustic_embedding(acoustic_ids)

        # First Channel
        visual_ = self.hv(text_embedding, visual_)
        acoustic_ = self.ha(text_embedding, acoustic_)
        visual_acoustic = torch.cat((visual_, acoustic_), dim=-1)
        F1 = self.cat_connect(visual_acoustic)

        # norm & add
        h1 = self.norm(F1)
        F1 = (F1 * h1) + F1
        F1 = self.ffn(F1)

        # NIRM
        v = self.hv(visual_, acoustic_)
        a = self.ha(acoustic_, visual_)
        v_a = torch.cat((v, a), dim=-1)
        F2 = self.cat_connect(v_a)

        # norm & add
        h2 = self.norm(F2)
        F2 = (F2 * h2) + F2
        F2 = self.ffn(F2)

        # TIIM
        F3 = self.ht(F1, text_embedding)

        # norm & add
        h3 = self.norm(F3)
        F3 = (F3 * h3) + F3
        F3 = self.ffn(F3)

        # skip-connection
        embedding_shift = text_embedding + F1 + F2 + F3

        # norm & add
        h = self.norm(embedding_shift)
        embedding_shift = (embedding_shift * h) + embedding_shift

        return embedding_shift

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, head_num=1):
        super(SelfAttention, self).__init__()
        self.head_num = head_num
        self.s_d = hidden_size // self.head_num
        self.all_head_size = self.head_num * self.s_d
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        x = x.view(x.size(0), x.size(1), self.head_num, -1)
        return x.permute(0, 2, 1, 3)

    def sinusoidal_position_embedding(self, batch_size, nums_head, max_len, output_dim, device):

        # (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)

        # (output_dim//2)
        ids = torch.arange(0, output_dim // 2, dtype=torch.float)
        theta = torch.pow(10000, -2 * ids / output_dim)

        # (max_len, output_dim//2)
        embeddings = position * theta

        # (max_len, output_dim//2, 2)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # (bs, head, max_len, output_dim//2, 2)
        embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))

        # (bs, head, max_len, output_dim)
        embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
        embeddings = embeddings.to(device)

        return embeddings

    def RoPE(self, q, k):
        # q,k: (bs, head, max_len, output_dim)
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        batch_size = q.shape[0]
        nums_head = q.shape[1]
        max_len = q.shape[2]
        output_dim = q.shape[-1]

        # (bs, head, max_len, output_dim)
        pos_emb = self.sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)

        # cos_pos,sin_pos: (bs, head, max_len, output_dim)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

        # q,k: (bs, head, max_len, output_dim)
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
        q2 = q2.reshape(q.shape)

        # Update qw and multiply the corresponding positions by *
        q = q * cos_pos + q2 * sin_pos

        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
        k2 = k2.reshape(k.shape)

        # Update kw, * multiply corresponding positions
        k = k * cos_pos + k2 * sin_pos

        return q.squeeze(1), k.squeeze(1)

    def forward(self, text_embedding, embedding):
        Q = self.Wq(text_embedding)
        K = self.Wk(embedding)

        # positional embedding
        Q, K = self.RoPE(Q, K)

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

# Feedforward neural network
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

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads=8, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hidden_size
        self.n_heads = n_heads

        assert hidden_size % n_heads == 0
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.do = nn.Dropout(dropout)

    def sinusoidal_position_embedding(self, batch_size, nums_head, max_len, output_dim, device):
        # (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)

        # (output_dim//2)
        ids = torch.arange(0, output_dim // 2, dtype=torch.float)
        theta = torch.pow(10000, -2 * ids / output_dim)

        # (max_len, output_dim//2)
        embeddings = position * theta

        # (max_len, output_dim//2, 2)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # (bs, head, max_len, output_dim//2, 2)
        embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))

        # (bs, head, max_len, output_dim)
        embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
        embeddings = embeddings.to(device)

        return embeddings

    def RoPE(self, q, k):
        # q,k: (bs, head, max_len, output_dim)
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        batch_size = q.shape[0]
        nums_head = q.shape[1]
        max_len = q.shape[2]
        output_dim = q.shape[-1]

        # (bs, head, max_len, output_dim)
        pos_emb = self.sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)

        # cos_pos,sin_pos: (bs, head, max_len, output_dim)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

        # q,k: (bs, head, max_len, output_dim)
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
        q2 = q2.reshape(q.shape)

        q = q * cos_pos + q2 * sin_pos

        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
        k2 = k2.reshape(k.shape)

        k = k * cos_pos + k2 * sin_pos

        return q.squeeze(1), k.squeeze(1)

    def forward(self, text_embedding, embedding, mask=None):
        bsz = text_embedding.shape[0]
        Q = self.w_q(text_embedding)
        K = self.w_k(embedding)
        Q, K = self.RoPE(Q, K)

        V = self.w_v(embedding)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2))

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        attention = self.do(torch.softmax(attention * 8, dim=-1))

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)

        return x






