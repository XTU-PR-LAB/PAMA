import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        """
            Scaled Dot-Product Attention
            out = softmax(QK^T/temperature)V
        :param q: (bs, lenq, d_k)
        :param k: (bs, lenv, d_k)
        :param v: (bs, lenv, d_v)
        :param mask: None
        :return:
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """
        Multi-Head Attention module
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)  #Linear(256+64,8*32)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)  #Linear(8*32,256+64)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)  #根号d_k

        self.dropout = nn.Dropout(dropout)
        # self.bn = nn.BatchNorm1d(d_model)
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):
        """
        :param q: (bs, lenq, d_model)
        :param k: (bs, lenv, d_model)
        :param v: (bs, lenv, d_model)
        :param mask:
        :return: (bs, lenq, d_model)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        bs, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: bs x lq x (n*dv)
        # Separate different heads: bs x lq x n x dv
        q = self.w_qs(q).view(bs, len_q, n_head, d_k)  #(B*N,V,n_head,d_k)
        k = self.w_ks(k).view(bs, len_k, n_head, d_k)
        v = self.w_vs(v).view(bs, len_v, n_head, d_v)

        # Transpose for attention dot product: bs x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  #(B*N,n_head,V,d_k)
        q, attn = self.attention(q, k, v)  #q:(B*N,n_head,V,d_k),atten:(B*N,n_head,V,V)

        # Transpose to move the head dimension back: bs x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b sx lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(bs, len_q, -1)  #(B*N,V,n_head*d_k)
        q = self.dropout(self.fc(q))  #(B*N,V,C)
        q += residual
        # q = self.bn(q.permute((0, 2, 1))).permute((0, 2, 1))
        # q = self.layer_norm(q)
        if self.norm:
            q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1, norm=False):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.bn = nn.BatchNorm1d(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        if self.norm:
        # x = self.bn(x.permute((0, 2, 1))).permute((0, 2, 1))
            x = self.layer_norm(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class EncoderLayer(nn.Module):
    """
        multi-attention + position feed forward
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, norm=False):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, norm=norm)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, norm=norm)

    def forward(self, enc_input): 
        enc_output, enc_slf_attn = self.attn(  #enc_input:(B,2,C)
            enc_input[:,0:1,:], enc_input[:,1:2,:], enc_input[:,1:2,:])

        enc_output = self.pos_ffn(enc_output)   #(B*N,V,C)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """ Compose with three layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, norm=False):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, norm=norm)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, norm=norm)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, norm=norm)

    def forward(self, dec_input, enc_output):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, norm=False, n_position=200):

        super().__init__()

        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, norm=norm)
            for _ in range(n_layers)])
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.bn = nn.BatchNorm1d(d_model)

    def forward(self, feats_embedding, return_attns=False):
        """
        :param feats_embedding: (bs, num_views, dim)
        :param return_attns:
        :return:
        """
        enc_slf_attn_list = []
        # enc_output = self.dropout(self.position_enc(feats_embedding))
        enc_output = feats_embedding  #(B*N,V,C)
        if self.norm:
            enc_output = self.layer_norm(enc_output)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1, norm=False, n_position=200):

        super().__init__()
        # self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, norm=norm)
            for _ in range(n_layers)])
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.bn = nn.BatchNorm1d(d_model)

    def forward(self, dec_input, enc_output, return_attns=False):
        """
        :param dec_input: (bs, 1, dim)
        :param enc_output:  (bs, num_views, dim)
        :param return_attns:
        :return:
        """
        dec_output = dec_input
        # dec_output = self.dropout(self.position_enc(dec_output))
        if self.norm:
            dec_output = self.layer_norm(dec_output)
        dec_slf_attn_list, dec_enc_attn_list = [], []
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1):

        super().__init__()
        self.encoder = Encoder(d_model=d_model, d_inner=d_inner, n_layers=n_layers,
                               n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.decoder = Decoder(d_model=d_model, d_inner=d_inner, n_layers=n_layers,
                               n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_feats, trg_feats):
        """
        :param src_feats: (bs, num_views, dim)
        :param trg_feats: (bs, 1, dim)
        :return: fused feats: (bs, 1, dim)
        """
        enc_output, *_ = self.encoder(src_feats)
        dec_output, *_ = self.decoder(trg_feats, enc_output)
        return dec_output