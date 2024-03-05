import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from copy import deepcopy
from simVP_cnn import Encoder
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import warnings
warnings.filterwarnings('ignore')


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

def runmean(data, n_run):
    ll = data.shape[0]
    data_run = np.zeros([ll])
    for i in range(ll):
        if i < (n_run - 1):
            data_run[i] = np.nanmean(data[0 : i + 1])
        else:
            data_run[i] = np.nanmean(data[i - n_run + 1 : i + 1])
    return data_run


def cal_ninoskill2(pre_nino_all, real_nino):
    """
    :param pre_nino_all: [n_yr,start_mon,lead_max]
    :param real_nino: [n_yr,12]
    :return: nino_skill: [12,lead_max]
    """
    lead_max = pre_nino_all.shape[2]
    nino_skill = np.zeros([12, lead_max])
    for ll in range(lead_max):
        lead = ll + 1
        dd = deepcopy(pre_nino_all[:, :, ll])
        for mm in range(12):
            bb = dd[:, mm]
            st_m = mm + 1
            terget = st_m + lead
            if 12 < terget <= 24:
                terget = terget - 12
            elif terget > 24:
                terget = terget - 24
            aa = deepcopy(real_nino[:, terget - 1])
            nino_skill[mm, ll] = np.corrcoef(aa, bb)[0, 1]
    return nino_skill


class make_embedding(nn.Module):
    def __init__(
        self,
        cube_dim,
        d_size,
        emb_spatial_size,
        max_len,
        device,
    ):
        """
        :param cube_dim: The number of grids in one patch cube
        :param d_size: the embedding length
        :param emb_spatial_size:The number of patches decomposed in a field, S
        :param max_len: look back or prediction length, T
        """
        super().__init__()
        # 1. temporal embedding
        pe = torch.zeros(max_len, d_size)
        temp_position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_size, 2) * -(np.log(10000.0) / d_size))
        pe[:, 0::2] = torch.sin(temp_position * div_term)
        pe[:, 1::2] = torch.cos(temp_position * div_term)
        self.pe_time = pe[None, None].to(device)
        # 2. spatial embedding
        self.spatial_pos = torch.arange(emb_spatial_size)[None, :, None].to(device)
        self.emb_space = nn.Embedding(emb_spatial_size, d_size)
        self.linear = nn.Linear(cube_dim, d_size)
        self.norm = nn.LayerNorm(d_size)

    def forward(self, x):
        assert len(x.size()) == 4
        embedded_space = self.emb_space(self.spatial_pos)
        x = self.linear(x) + self.pe_time[:, :, : x.size(2)] + embedded_space
        return self.norm(x)


def unfold_func(in_data, kernel_size):
    n_dim = len(in_data.size())
    assert n_dim == 4 or n_dim == 5
    data1 = in_data.unfold(-2, size=kernel_size[0], step=kernel_size[0])
    data1 = data1.unfold(-2, size=kernel_size[1], step=kernel_size[1]).flatten(-2)
    if n_dim == 4:
        data1 = data1.permute(0, 1, 4, 2, 3).flatten(1, 2)
    elif n_dim == 5:
        data1 = data1.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    assert data1.size(-3) == in_data.size(-3) * kernel_size[0] * kernel_size[1]
    return data1


def fold_func(tensor, output_size, kernel_size):
    tensor = tensor.float()
    n_dim = len(tensor.size())
    assert n_dim == 4 or n_dim == 5
    f = tensor.flatten(0, 1) if n_dim == 5 else tensor
    folded = F.fold(
        f.flatten(-2),
        output_size=output_size,
        kernel_size=kernel_size,
        stride=kernel_size,
    )
    if n_dim == 5:
        folded = folded.reshape(tensor.size(0), tensor.size(1), *folded.size()[1:])
    return folded


def clone_layer(layer_in, N):
    return nn.ModuleList([copy.deepcopy(layer_in) for _ in range(N)])


class layerConnect(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


def T_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    sc = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        assert mask.dtype == torch.bool
        assert len(mask.size()) == 2
        sc = sc.masked_fill(mask[None, None, None], float("-inf"))
    p_sc = F.softmax(sc, dim=-1)
    if dropout is not None:
        p_sc = dropout(p_sc)
    return torch.matmul(p_sc, value)


def S_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(
        query.transpose(2, 3), key.transpose(2, 3).transpose(-2, -1)
    ) / np.sqrt(d_k)
    p_sc = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_sc = dropout(p_sc)
    return torch.matmul(p_sc, value.transpose(2, 3)).transpose(2, 3)


class Attention(nn.Module):
    def __init__(
            self,
            dim=6,
            dim_head=18,  # 32
            dropout=0.,
            window_size=7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # myatt

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=window_height, w2=window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x=height, y=width)


class make_attention(nn.Module):
    def __init__(self, d_size, nheads, attention_module, dropout):
        super().__init__()
        assert d_size % nheads == 0
        self.d_k = d_size // nheads
        self.nheads = nheads
        self.linears = nn.ModuleList([nn.Linear(d_size, d_size) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.attention_module = attention_module

    def forward(self, query, key, value, mask=None):
        """
        Transform the query, key, value into different heads, then apply the attention in parallel
        Args:
            query, key, value: size (batch, S, T, d_size)
        Returns:
            (batch, S, T, d_size)
        """
        nbatches = query.size(0)
        nspace = query.size(1)
        ntime = query.size(2)
        query, key, value = [
            l(x)
            .view(x.size(0), x.size(1), x.size(2), self.nheads, self.d_k)
            .permute(0, 3, 1, 2, 4)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x = self.attention_module(query, key, value, mask=mask, dropout=self.dropout)
        x = (
            x.permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(nbatches, nspace, ntime, self.nheads * self.d_k)
        )
        return self.linears[-1](x)


class miniEncoder(nn.Module):
    def __init__(self, d_size, nheads, dim_feedforward, dropout, layer_dim=256, dim_head=256, w=1):
        super().__init__()
        self.sublayer = clone_layer(layerConnect(size=d_size, dropout=dropout), 2)
        self.time_attn = make_attention(d_size, nheads, T_attention, dropout)
        self.space_attn = nn.Sequential(
            # //
            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),  # block-like myatt
            PreNormResidual(layer_dim,
                            Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)),
            PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout)),
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

            Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),  # grid-like myatt
            PreNormResidual(layer_dim,
                            Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)),
            PreNormResidual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout)),
            Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
        )

        self.cnnLayer = Encoder(256, 256, 4)

        self.FC = nn.Sequential(
            nn.Linear(d_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_size),
        )

    def TS_attn(self, query, key, value, mask):
        """
        Returns:
            (batch, S, T, d_size)
        """
        tt = self.time_attn(query, key, value, mask)  # tt (B,H*W,S,C)
        H = 17
        W = 30
        B, _, S, C = tt.shape  # 2,510,12,256
        x = tt.permute(0, 2, 3, 1)
        x = torch.unsqueeze(x, -1)
        x = x.view(B, S, C, H, W)
        x = x.reshape(B * S, C, H, W)
        tt = self.space_attn(x)  # tt (B*S,C,H,W)
        tt = torch.unsqueeze(tt, 0)
        tt = tt.view(B, S, C, H, W)
        tt = tt.view(B, S, C, H * W)
        tt = tt.permute(0, 3, 1, 2)
        return tt

    def forward(self, x, mask=None):  # x (B,H*W,S,C)
        H = 17
        W = 30
        B, _, S, C = x.shape   # 1,510,12,256
        # 时空注意力分支
        x1 = self.sublayer[0](x, lambda x: self.TS_attn(x, x, x, mask))   # x1 (B,H*W,S,C)
        # 卷积分支
        # (1 * 12, 256, 17, 30)
        x = x.permute(0, 2, 3, 1)
        x = torch.unsqueeze(x, -1)
        x = x.view(B, S, C, H, W)
        x = x.reshape(B*S, C, H, W)
        x2 = self.cnnLayer(x)
        x2 = torch.unsqueeze(x2, 0)
        x2 = x2.view(B, S, C, H, W)
        x2 = x2.reshape(B, S, C, H*W)
        x2 = x2.permute(0, 3, 1, 2)

        # 定义加权平均的权重
        weight_method1 = 0.7  # 可以根据需要调整权重
        weight_method2 = 0.3
        combineX = weight_method1 * x1 + weight_method2 * x2
        return self.sublayer[1](combineX, self.FC)


class miniDecoder(nn.Module):
    def __init__(self, d_size, nheads, dim_feedforward, dropout):
        super().__init__()
        self.sublayer = clone_layer(layerConnect(d_size, dropout), 3)
        self.encoder_attn = make_attention(d_size, nheads, T_attention, dropout)
        self.time_attn = make_attention(d_size, nheads, T_attention, dropout)
        self.space_attn = make_attention(d_size, nheads, S_attention, dropout)
        self.FC = nn.Sequential(
            nn.Linear(d_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_size),
        )

    def divided_TS_attn(self, query, key, value, mask=None):
        m = self.time_attn(query, key, value, mask)
        return self.space_attn(m, m, m)

    def forward(self, x, en_out, tgt_mask, memory_mask):
        x = self.sublayer[0](x, lambda x: self.divided_TS_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](
            x, lambda x: self.encoder_attn(x, en_out, en_out, memory_mask)
        )
        return self.sublayer[2](x, self.FC)
