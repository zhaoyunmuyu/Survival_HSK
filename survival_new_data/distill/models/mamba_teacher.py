"""教师模型：Mamba-SSM 序列编码 + 静态特征融合（HSK 蒸馏迁移版）

迁移自 survival_kd，主要差异：
- TokenBuilder 维度可配置（默认对接 BERT 768 维文本向量）。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from survival_new_data.distill.models.fusion import TokenBuilder


class _FallbackEncoder(nn.Module):
    """当 Mamba 依赖缺失时，使用简单 Transformer 编码器回退。"""

    def __init__(self, d_model: int = 512, n_layers: int = 4, n_heads: int = 8, dim_feedforward: int = 1024):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class MambaTeacher(nn.Module):
    """教师模型（优先使用 Mamba-SSM）。"""

    def __init__(
        self,
        *,
        d_model: int = 512,
        n_layers: int = 4,
        dropout: float = 0.1,
        text_dim: int = 768,
        img_dim: int = 512,
        feature_dim: int = 8,
        time_step: TokenBuilder._TimeStep = "year",
        max_offset_years: int = 10,
        use_macro_features: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_macro_features = bool(use_macro_features)
        self.token_builder = TokenBuilder(
            d_model=self.d_model,
            text_dim=int(text_dim),
            img_dim=int(img_dim),
            feature_dim=int(feature_dim),
            time_step=time_step,
            max_offset_years=int(max_offset_years),
        )

        self.region_proj = nn.Linear(18, self.d_model)
        self.macro_proj = (
            nn.Sequential(
                nn.Linear(62, self.d_model),
                nn.ReLU(),
            )
            if self.use_macro_features
            else None
        )

        try:
            from mamba_ssm import Mamba  # type: ignore

            layers = []
            for _ in range(int(n_layers)):
                layers.append(Mamba(self.d_model))
            self.encoder = nn.Sequential(*layers)
            self._use_mamba = True
        except Exception:
            self.encoder = _FallbackEncoder(d_model=self.d_model)
            self._use_mamba = False

        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 1024),
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
        # 防御性处理：one_hot 在 CUDA 上遇到越界会触发 device-side assert
        region_idx = region_idx.clamp(min=0, max=17)
        region_onehot = F.one_hot(region_idx, num_classes=18).float()
        return self.region_proj(region_onehot)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).float()
        denom = valid.sum(dim=1).clamp(min=1.0)
        summed = (x * valid.unsqueeze(-1)).sum(dim=1)
        return summed / denom.unsqueeze(-1)

    def forward(self, batch: dict) -> torch.Tensor:
        tokens, padding_mask = self.token_builder(batch)

        # padding 位置的 token（尤其包含 time embedding）会干扰 Mamba 的顺序建模，
        # 需显式置零，并在每层后重新 mask。
        tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if self._use_mamba:
            x = tokens
            for layer in self.encoder:
                x = layer(x)
                x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        else:
            x = self.encoder(tokens, src_key_padding_mask=padding_mask)

        seq_feat = self._masked_mean(x, padding_mask)
        region_feat = self._region_feature(batch["region_encoding"])
        if self.use_macro_features and self.macro_proj is not None:
            macro = batch.get("macro_features")
            if macro is not None:
                macro = macro.float()
                if macro.dim() == 1:
                    macro = macro.unsqueeze(0)
                macro_feat = self.macro_proj(macro)
            else:
                macro_feat = torch.zeros((seq_feat.size(0), self.d_model), dtype=seq_feat.dtype, device=seq_feat.device)
        else:
            macro_feat = torch.zeros((seq_feat.size(0), self.d_model), dtype=seq_feat.dtype, device=seq_feat.device)
        fused = self.norm(self.dropout(seq_feat + region_feat + macro_feat))

        logits = self.classifier(fused)
        return logits
