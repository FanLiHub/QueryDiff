import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class StageGrouper(nn.Module):
    def __init__(
        self,
        r: int,
        c: int,
        L_in: int,
        L_out: int,
        embed_dim: int = None,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert L_out <= L_in, f"L_out ({L_out}) must be <= L_in ({L_in})"
        self.r = r
        self.c = c
        self.L_in = L_in
        self.L_out = L_out

        if embed_dim is None:
            embed_dim = r * c
        self.embed_dim = embed_dim

        self.proj_in = nn.Linear(r * c, embed_dim)

        self.group_tokens = nn.Parameter(torch.randn(L_out, embed_dim))

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.proj_out = nn.Linear(embed_dim, r * c)

    def forward(self, Q: Tensor) -> Tuple[Tensor, Tensor]:

        assert Q.ndim == 3, f"Expected Q.shape == (L, r, c), got {Q.shape}"
        L_in, r, c = Q.shape
        assert L_in == self.L_in, f"Expected L_in={self.L_in}, got {L_in}"
        assert r == self.r and c == self.c, f"Expected (r,c)=({self.r},{self.c}), got ({r},{c})"

        B = 1
        x = Q.reshape(1, self.L_in, self.r * self.c)
        x = self.proj_in(x)

        group = self.group_tokens.unsqueeze(0)

        out, attn = self.attn(query=group, key=x, value=x)

        out = self.norm(out)

        out_flat = self.proj_out(out)
        Q_out = out_flat.reshape(self.L_out, self.r, self.c)

        return Q_out, attn


class FinalQueryHead(nn.Module):

    def __init__(self, c: int, mlp_hidden_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(c * mlp_hidden_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, c),
        )

    def forward(self, Q_last: Tensor) -> Tensor:

        assert Q_last.ndim == 3
        q_mean = Q_last.mean(dim=0)
        q_out = self.mlp(q_mean)
        return q_out


class HierarchicalQueryEncoder(nn.Module):
    def __init__(
        self,
        r: int,
        c: int,
        L_list: List[int],
        embed_dim: int = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_hidden_ratio: float = 4.0,
    ):
        super().__init__()
        assert len(L_list) == 4
        L0, L1, L2, L3 = L_list
        assert L0 >= L1 >= L2 >= L3

        self.r = r
        self.c = c
        self.L_list = L_list

        self.stage1 = StageGrouper(r=r, c=c, L_in=L0, L_out=L1,
                                   embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.stage2 = StageGrouper(r=r, c=c, L_in=L1, L_out=L2,
                                   embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.stage3 = StageGrouper(r=r, c=c, L_in=L2, L_out=L3,
                                   embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        self.final_head = FinalQueryHead(c=c, mlp_hidden_ratio=mlp_hidden_ratio)

    def forward(self, Q0: Tensor):
        assert Q0.ndim == 3
        L0, r, c = Q0.shape
        assert L0 == self.L_list[0]
        assert r == self.r and c == self.c, f"Expected (r,c)=({self.r},{self.c}), got ({r},{c})"

        all_Q = [Q0]

        Q1, _ = self.stage1(Q0)
        all_Q.append(Q1)

        Q2, _ = self.stage2(Q1)
        all_Q.append(Q2)

        Q3, _ = self.stage3(Q2)
        all_Q.append(Q3)

        Q_single = self.final_head(Q3)

        return Q_single, all_Q
