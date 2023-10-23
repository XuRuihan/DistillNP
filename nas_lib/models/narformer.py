import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple

import math
import algos

from torch import Tensor


def attention(query, key, value, dropout=None, rel_pos_bias=None):
    d_k = query.size(-1)
    # (b, n_head, l_q, d_per_head) * (b, n_head, d_per_head, l_k)
    attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if rel_pos_bias is not None:
        attn = attn + rel_pos_bias
        # attn = attn.masked_fill(rel_pos_bias==8, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    if dropout is not None:
        attn = dropout(attn)  # (b, n_head, l_q, l_k)
    return torch.matmul(attn, value)


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_head, dropout, q_learnable, rel_pos_bias=False):
        super().__init__()
        self.q_learnable = q_learnable
        self.dim = dim
        self.n_head = n_head
        self.d_k = dim // n_head  # default: 32

        self.linears = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])
        if q_learnable:
            self.linears.append(nn.Identity())
        else:
            self.linears.append(nn.Linear(dim, dim))

        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rel_pos_bias = rel_pos_bias
        if rel_pos_bias:
            self.rel_pos_encoder_forward = nn.Embedding(10, self.n_head, padding_idx=9)
            self.rel_pos_encoder_backward = nn.Embedding(10, self.n_head, padding_idx=9)

    def forward(self, query, key, value, rel_pos=None):
        batch_size = query.size(0)

        key, value, query = [
            l(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (key, value, query))
        ]
        if self.rel_pos_bias:
            rel_pos_bias = (
                self.rel_pos_encoder_forward(rel_pos)
                + self.rel_pos_encoder_backward(rel_pos.transpose(-2, -1))
            ).permute(0, 3, 1, 2)
        else:
            rel_pos_bias = None
        # x: (b, n_head, l_q, d_k), attn: (b, n_head, l_q, l_k)
        x = attention(
            query, key, value, dropout=self.attn_dropout, rel_pos_bias=rel_pos_bias
        )
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        return self.resid_dropout(self.proj(x))


# Different Attention Blocks, All Based on MultiHeadAttention
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, n_head, dropout, droppath, rel_pos_bias=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim,
            n_head,
            dropout,
            q_learnable=False,
            rel_pos_bias=rel_pos_bias,
        )
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()

    def forward(self, x, rel_pos):
        x_ = self.norm(x)
        x_ = self.attn(x_, x_, x_, rel_pos)
        return self.drop_path(x_) + x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, n_head, dropout, droppath):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_head, dropout, q_learnable=True)
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()

    def forward(self, x, learnt_q):
        x_ = self.norm(x)
        x_ = self.attn(learnt_q, x_, x_)
        # In multi_stage' attention, no residual connection is used because of the change in output shape
        return self.drop_path(x_)


class Mlp(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4,
        out_features=None,
        act_layer="relu",
        drop=0.0,
        bias=True,
        gcn=False,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)
        self.gcn = gcn

        if gcn:
            # self.fc0 = nn.Linear(in_features, hidden_features, bias=bias)
            # self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.fc1 = nn.Linear(in_features * 2, hidden_features, bias=bias)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        if act_layer.lower() == "relu":
            self.act = nn.ReLU()
        elif act_layer.lower() == "leaky_relu":
            self.act = nn.LeakyReLU()
        elif act_layer.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, L):
        if self.gcn:
            x = torch.cat([x, torch.matmul(L, x)], dim=-1)
            # x = torch.cat([x, torch.matmul(L.transpose(-2, -1), x)], dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, act_layer, dropout, droppath, gcn=False):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.feed_forward = Mlp(
            dim, mlp_ratio, act_layer=act_layer, drop=dropout, gcn=gcn
        )
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()

    def forward(self, x, L=None):
        x_ = self.norm(x)
        x_ = self.feed_forward(x_, L)
        return self.drop_path(x_) + x


class EncoderBlock(nn.Module):
    def __init__(self, dim, n_head, mlp_ratio, act_layer, dropout, droppath):
        super().__init__()
        self.self_attn = SelfAttentionBlock(
            dim,
            n_head,
            dropout,
            droppath,
            rel_pos_bias=True,
        )
        self.feed_forward = FeedForwardBlock(
            dim, mlp_ratio, act_layer, dropout, droppath, gcn=True
        )

    def forward(self, x, rel_pos, L):
        x = self.self_attn(x, rel_pos)
        x = self.feed_forward(x, L)
        return x


# Blocks Used in Encoder
class FuseFeatureBlock(nn.Module):
    def __init__(self, dim, n_head, dropout, droppath, mlp_ratio=4, act_layer="relu"):
        super().__init__()
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.fuse_attn = MultiHeadAttention(dim, n_head, dropout, q_learnable=False)
        self.feed_forward = FeedForwardBlock(
            dim, mlp_ratio, act_layer, dropout, droppath
        )

    def forward(self, memory, q):
        x_ = self.norm_kv(memory)
        q_ = self.norm_q(q)
        x = self.fuse_attn(q_, x_, x_)
        x = self.feed_forward(x)
        return x


class FuseStageBlock(nn.Module):
    def __init__(
        self,
        depths,
        dim,
        n_head,
        mlp_ratio,
        act_layer,
        dropout,
        droppath,
        stg_id,
        dp_rates,
    ):
        super().__init__()
        self.n_self_attn = depths[stg_id] - 1
        self.self_attns = nn.ModuleList()
        self.feed_forwards = nn.ModuleList()
        for i, droppath in enumerate(dp_rates):
            if i == 0:
                self.cross_attn = CrossAttentionBlock(
                    dim,
                    n_head,
                    dropout,
                    droppath,
                )
            else:
                self.self_attns.append(
                    SelfAttentionBlock(
                        dim,
                        n_head,
                        dropout,
                        droppath,
                    )
                )
            self.feed_forwards.append(
                FeedForwardBlock(dim, mlp_ratio, act_layer, dropout, droppath)
            )

    def forward(self, kv, q):
        x = self.cross_attn(kv, q)
        x = self.feed_forwards[0](x)
        for i in range(self.n_self_attn):
            x = self.self_attns[i](x)
            x = self.feed_forwards[i + 1](x)
        return x


# Main class
class Encoder(nn.Module):
    def __init__(
        self,
        depths=[6, 1, 1, 1],
        dim=192,
        n_head=6,
        mlp_ratio=4,
        act_layer="relu",
        dropout=0.1,
        droppath=0.0,
    ):
        super().__init__()
        self.num_stage = len(depths)
        self.num_layers = sum(depths)
        self.norm = nn.LayerNorm(dim)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, droppath, self.num_layers)]

        # 1st stage: Encoder
        self.layers = nn.ModuleList()
        for i in range(depths[0]):
            droppath = dpr[i]
            self.layers.append(
                EncoderBlock(dim, n_head, mlp_ratio, act_layer, dropout, droppath)
            )

        if self.num_stage > 1:
            # Rest stage: information fusion
            self.fuseUnit = nn.ModuleList()
            self.fuseStages = nn.ModuleList()
            self.fuseStages.append(
                FuseStageBlock(
                    depths,
                    dim,
                    n_head,
                    mlp_ratio,
                    act_layer,
                    dropout,
                    droppath,
                    stg_id=1,
                    dp_rates=dpr[sum(depths[:1]) : sum(depths[:2])],
                )
            )
            for i in range(2, self.num_stage):
                self.fuseUnit.append(
                    FuseFeatureBlock(
                        dim,
                        n_head,
                        dropout,
                        droppath,
                        mlp_ratio,
                        act_layer,
                    )
                )
                self.fuseStages.append(
                    FuseStageBlock(
                        depths,
                        dim,
                        n_head,
                        mlp_ratio,
                        act_layer,
                        dropout,
                        droppath,
                        stg_id=i,
                        dp_rates=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    )
                )

            self.learnt_q = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(1, 2 ** (3 - s), dim))
                    for s in range(1, self.num_stage)
                ]
            )

    def forward(self, x, rel_pos, adj):
        B, _, _ = x.shape

        # 1st stage: Encoder
        for i, layer in enumerate(self.layers):
            x = layer(x, rel_pos, adj)
        x_ = x
        # Rest stage: information fusion
        if self.num_stage > 1:
            memory = x
            q = self.fuseStages[0](
                memory, self.learnt_q[0].repeat(B, 1, 1, 1)
            )  # q(b,4,d)
            for i in range(self.num_stage - 2):
                kv = self.fuseUnit[i](memory, q)
                q = self.fuseStages[i + 1](
                    kv, self.learnt_q[i + 1].repeat(B, 1, 1, 1)
                )  # q(b,2,d), q(b,1,d)
            x_ = q
        output = self.norm(x_)
        return output


class RegHead(nn.Module):
    def __init__(self, in_channels, avg_tokens, out_channels=1):
        super().__init__()
        self.avg_tokens = avg_tokens
        self.layer = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:  # x(b/n_gpu, l, d)
        if self.avg_tokens:
            x_ = x.mean(dim=1)
        else:
            x_ = x[:, 0, :]  # (b, d)

        res = self.layer(x_)
        return res


class Embedder:
    def __init__(self, num_freqs, embed_type="mine", input_type="tensor", input_dims=1):
        self.num_freqs = num_freqs
        self.max_freq = max(32, num_freqs)
        self.embed_type = embed_type
        self.input_type = input_type
        self.input_dims = input_dims
        self.eps = 1e-2
        if input_type == "tensor":
            self.embed_fns = [torch.sin, torch.cos]
            self.embed = self.embed_tensor
        else:
            self.embed_fns = [np.sin, np.cos]
            self.embed = self.embed_array
        self.create_embedding_fn()

    def __call__(self, x):
        return self.embed(x)

    def create_embedding_fn(self):
        max_freq = self.max_freq
        N_freqs = self.num_freqs

        freq_bands = (
            (self.eps + torch.linspace(1, max_freq, N_freqs)) * math.pi / (max_freq + 1)
        )

        self.freq_bands = freq_bands
        self.out_dim = self.input_dims * len(self.embed_fns) * len(freq_bands)

    def embed_tensor(self, inputs):
        self.freq_bands = self.freq_bands.to(inputs.device)
        return torch.cat([fn(self.freq_bands * inputs) for fn in self.embed_fns], -1)

    def embed_array(self, inputs):
        return np.concatenate([fn(self.freq_bands * inputs) for fn in self.embed_fns])


# original
def tokenizer(ops, adj: Tensor, dim_x=32, dim_r=32, dim_p=32, embed_type="mine"):
    assert dim_x == dim_p == dim_r
    dim = dim_x + dim_p + dim_r
    adj = torch.tensor(adj)

    # encode operation
    fn = Embedder(dim_x, embed_type=embed_type)
    code_ops_list = [fn(torch.Tensor([30]))]
    code_ops_list += [fn(torch.Tensor([op])) for op in ops]
    code_ops = torch.stack(code_ops_list, dim=0)  # (len, dim_x)

    # encode self position
    code_pos_list = [fn(torch.Tensor([30]))]
    code_pos_list += [fn(torch.Tensor([i])) for i in range(len(ops))]
    code_pos = torch.stack(code_pos_list, dim=0)  # (len, dim_p)

    # encode in-degree nodes
    c_adj = torch.eye(len(ops) + 1)
    c_adj[1:, 1:] = torch.Tensor(adj)
    code_sour = c_adj.T @ code_pos
    code_sour[0] = fn(torch.Tensor([30]))
    code_sour[1] = fn(torch.Tensor([-1]))

    code = torch.cat([code_ops, code_pos, code_sour], dim=-1)

    depth = torch.Tensor([len(ops)])
    depth_fn = Embedder(dim, embed_type=embed_type)
    code_depth = depth_fn(depth).reshape(1, -1)

    shortest_path, path = algos.floyd_warshall(adj.numpy())
    shortest_path = torch.from_numpy(shortest_path).long()
    shortest_path = torch.clamp(shortest_path, min=0, max=8)

    rel_pos = torch.full((len(ops) + 2, len(ops) + 2), fill_value=9).int()
    rel_pos[1:-1, 1:-1] = shortest_path

    c_adj_d = torch.zeros((len(ops) + 2, len(ops) + 2)).int()
    c_adj_d[1:-1, 1:-1] = adj

    return code, rel_pos, c_adj_d, code_depth


class NARFormer(nn.Module):
    def __init__(
        self,
        depths=[6, 1, 1, 1],
        dim=192,
        n_head=6,
        mlp_ratio=4,
        act_layer="relu",
        dropout=0.1,
        droppath=0.0,
        avg_tokens=False,
        use_extra_token=True,
        out_dim=1,
    ):
        super().__init__()
        self.dim = dim
        self.use_extra_token = use_extra_token

        self.transformer = Encoder(
            depths=depths,
            dim=dim,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            dropout=dropout,
            droppath=droppath,
        )
        self.head = RegHead(dim, avg_tokens, out_dim)
        self.dep_map = nn.Linear(dim, dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.constant_(m.weight, 0)
            # nn.init.trunc_normal_(m.weight, std=0.02)

    @torch.jit.ignore()
    def no_weight_decay(self):
        no_decay = {}
        return no_decay

    def forward(self, seqcode, rel_pos, adjs, code_depth):
        # Depth token
        code_depth = F.relu(self.dep_map(code_depth))
        seqcode = torch.cat([seqcode, code_depth], dim=1)

        aev = self.transformer(seqcode, rel_pos, adjs.to(torch.float))
        # multi_stage:aev(b, 1, d)
        predict = self.head(aev)
        return predict


class NARFormerLoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_rank=0.2, lambda_consist=0.0):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_rank = lambda_rank
        self.lambda_consist = lambda_consist
        self.loss_mse = nn.MSELoss()
        self.loss_rank = nn.L1Loss()
        self.loss_consist = nn.L1Loss()

    def forward(self, predict: Tensor, target: Tensor) -> Tensor:
        loss_mse = self.loss_mse(predict, target) * self.lambda_mse

        index = torch.randperm(predict.shape[0], device=predict.device)
        v1 = predict - predict[index]
        v2 = target - target[index]
        loss_rank = self.loss_rank(v1, v2) * self.lambda_rank

        loss_consist = 0
        if self.lambda_consist > 0:
            source_pred, aug_pred = predict.chunk(2, 0)
            loss_consist = (
                self.loss_consist(source_pred, aug_pred) * self.lambda_consist
            )
        loss = loss_mse + loss_rank + loss_consist
        return {
            "loss": loss,
            "loss_mse": loss_mse,
            "loss_rank": loss_rank,
            "loss_consist": loss_consist,
        }
