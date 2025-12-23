"""注意力模块集合

包含：
- FactorAtt：将序列特征与外部辅助特征（如评论属性/宏观数据）耦合的注意力；
- TransformerBlock：标准自注意力 + 前馈 + 残差规范；
- CrossAttentionBlock：跨模态注意力，Q 来自一侧，K/V 来自另一侧。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorAtt(nn.Module):
    """结合外部因子的序列注意力。

    思路：
    - 先对输入序列做 Q/K/V 线性变换；
    - 依据 K 的自相关计算注意力底稿 r（bet）；
    - 使用外部特征（features）做列 softmax 后求和，得到位置权重 extra_para；
    - 将二者相乘得到最终注意力矩阵，作用于 V，输出加权后的序列。
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.d_k = feature_dim
        self.Wq = nn.Linear(self.d_k, self.d_k)
        nn.init.normal_(self.Wq.weight, 0, 0.1)
        self.Wk = nn.Linear(self.d_k, self.d_k)
        nn.init.normal_(self.Wk.weight, 0, 0.1)
        self.Wv = nn.Linear(self.d_k, self.d_k)
        nn.init.normal_(self.Wv.weight, 0, 0.1)

    def forward(self, H_T: torch.Tensor, features: torch.Tensor, bias: float) -> torch.Tensor:
        seq_len = H_T.size(0)
        q = self.Wq(H_T)
        k = self.Wk(H_T)
        v = self.Wv(H_T)

        r = self.compute_r(seq_len, k)
        alpha = self.compute_alpha(seq_len, r, features, bias)
        return torch.matmul(alpha, v)

    def compute_r(self, seq_len: int, k: torch.Tensor) -> torch.Tensor:
        """计算 K 的两两相似度（缩放），作为时间位置间的相关性。"""
        return (k @ k.T) / np.sqrt(self.d_k)

    def compute_alpha(self, seq_len: int, bet: torch.Tensor, features: torch.Tensor, bias: float) -> torch.Tensor:
        """融合外部因子，得到最终的位置权重。"""
        soft_bet = bet  # 此处保留 softmax 前的形式，按原实现保留
        features_with_bias = features + bias
        col_softmax = F.softmax(features_with_bias, dim=0)
        extra_para = col_softmax.sum(dim=1) + 1e-8
        if torch.sum(torch.abs(extra_para)) < 1e-8:
            extra_para = torch.ones_like(extra_para)
        extra_para = extra_para.view(seq_len, -1)
        return extra_para * soft_bet


class TransformerBlock(nn.Module):
    """标准自注意力块：自注意力 + 残差 + FFN + 归一化。"""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w1 = nn.Parameter(torch.ones(embed_dim))
        self.w2 = nn.Parameter(torch.ones(embed_dim))
        self.w3 = nn.Parameter(torch.ones(embed_dim))
        self.w4 = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(self.w1 * x + self.w2 * self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(self.w3 * x + self.w4 * self.dropout2(ffn_output))
        return x


class CrossAttentionBlock(nn.Module):
    """跨模态注意力块：Q 与 K/V 可以来自不同模态。"""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w1 = nn.Parameter(torch.ones(embed_dim))
        self.w2 = nn.Parameter(torch.ones(embed_dim))
        self.w3 = nn.Parameter(torch.ones(embed_dim))
        self.w4 = nn.Parameter(torch.ones(embed_dim))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_output, _ = self.cross_attn(q, k, v, key_padding_mask=key_padding_mask)
        x = self.norm1(self.w1 * q + self.w2 * self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(self.w3 * x + self.w4 * self.dropout2(ffn_output))
        return x
