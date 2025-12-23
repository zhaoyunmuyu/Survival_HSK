"""教师模型：Mamba-SSM 序列编码 + 静态特征融合

说明：
- 优先使用 mamba-ssm 提供的 Mamba 块；若依赖不可用则回退为 TransformerEncoder；
- 输入为评论 token 序列（来自 TokenBuilder）以及区域/宏观静态特征；
- 输出二分类 logits（B,1）。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from survival_kd.models.fusion import TokenBuilder


def _build_region_feature(region_encoding: torch.Tensor, d_model: int) -> torch.Tensor:
    """区域编码（1..18）→ one-hot → 线性映射到 d_model。"""
    region_idx = region_encoding.squeeze().long() - 1
    region_onehot = F.one_hot(region_idx.clamp(min=0), num_classes=18).float()
    proj = nn.Linear(18, d_model).to(region_onehot.device)
    # 为了在 forward 中使用，需要将权重注册至模块；此处在类中实现。
    raise RuntimeError("This helper should not be called directly")


class _FallbackEncoder(nn.Module):
    """当 Mamba 依赖缺失时，使用简单 Transformer 编码器回退。"""

    def __init__(self, d_model: int = 512, n_layers: int = 4, n_heads: int = 8, dim_feedforward: int = 1024):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class MambaTeacher(nn.Module):
    """教师模型（优先使用 Mamba-SSM）。"""

    def __init__(self, d_model: int = 512, n_layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_builder = TokenBuilder(d_model=d_model)

        # 区域与宏观的静态映射
        self.region_proj = nn.Linear(18, d_model)
        self.macro_proj = nn.Sequential(
            nn.Linear(62, d_model),
            nn.ReLU(),
        )

        # 优先尝试导入 Mamba，否则回退 Transformer
        try:
            from mamba_ssm import Mamba  # type: ignore

            layers = []
            for _ in range(n_layers):
                layers.append(Mamba(d_model))
            self.encoder = nn.Sequential(*layers)
            self._use_mamba = True
        except Exception:
            self.encoder = _FallbackEncoder(d_model=d_model)
            self._use_mamba = False

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1),
        )

    def _region_feature(self, region_encoding: torch.Tensor) -> torch.Tensor:
        region_idx = region_encoding.squeeze().long() - 1
        region_onehot = F.one_hot(region_idx.clamp(min=0), num_classes=18).float()
        return self.region_proj(region_onehot)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """按 mask=False 的有效位置做均值池化。"""
        # mask: True=padding；有效位 = ~mask
        valid = (~mask).float()
        denom = valid.sum(dim=1).clamp(min=1.0)  # [B]
        summed = (x * valid.unsqueeze(-1)).sum(dim=1)  # [B, d]
        return summed / denom.unsqueeze(-1)

    def forward(self, batch: dict) -> torch.Tensor:
        # 序列 token 与 padding 掩码
        tokens, padding_mask = self.token_builder(batch)  # [B,L,d], [B,L]

        if self._use_mamba:
            # Mamba 顺序处理（不支持 key_padding_mask，直接让 padding 位为 0 的 token 自然衰减）
            x = tokens
            for layer in self.encoder:
                x = layer(x)
        else:
            x = self.encoder(tokens, src_key_padding_mask=padding_mask)

        # 池化
        seq_feat = self._masked_mean(x, padding_mask)

        # 静态特征融合
        region_feat = self._region_feature(batch["region_encoding"])  # [B,d]
        macro_feat = self.macro_proj(batch["macro_features"])  # [B,62]->[B,d]
        fused = self.norm(self.dropout(seq_feat + region_feat + macro_feat))

        logits = self.classifier(fused)
        return logits
