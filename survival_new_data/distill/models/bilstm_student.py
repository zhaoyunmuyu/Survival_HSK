"""学生模型：BiLSTM 序列编码 + 静态特征融合（仅“最后两年”）

迁移自 survival_kd，主要差异：
- TokenBuilder 维度可配置（默认对接 BERT 768 维文本向量）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from survival_new_data.distill.models.fusion import TokenBuilder


class BiLSTMStudent(nn.Module):
    """学生模型：双向 LSTM 序列编码。"""

    def __init__(
        self,
        *,
        d_model: int = 512,
        num_layers: int = 2,
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

        hidden = self.d_model // 2
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=hidden,
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout),
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

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).float()
        denom = valid.sum(dim=1).clamp(min=1.0)
        summed = (x * valid.unsqueeze(-1)).sum(dim=1)
        return summed / denom.unsqueeze(-1)

    def _region_feature(self, region_encoding: torch.Tensor) -> torch.Tensor:
        region_idx = region_encoding.squeeze().long() - 1
        # 防御性处理：one_hot 在 CUDA 上遇到越界会触发 device-side assert
        region_idx = region_idx.clamp(min=0, max=17)
        region_onehot = F.one_hot(region_idx, num_classes=18).float()
        return self.region_proj(region_onehot)

    def forward(self, batch: dict) -> torch.Tensor:
        tokens, padding_mask = self.token_builder(batch)

        last2_mask = batch["last2_mask"].to(torch.bool)  # True 表示“属于最后两年”
        effective_padding = padding_mask | (~last2_mask)

        # LSTM 不支持 key_padding_mask：必须将被 mask 的位置置零，
        # 避免被“丢弃”的 token 通过隐藏状态影响后续 token（尤其是 last-window 之前的序列）。
        tokens = tokens.masked_fill(effective_padding.unsqueeze(-1), 0.0)

        lstm_out, _ = self.lstm(tokens)
        seq_feat = self._masked_mean(lstm_out, effective_padding)

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
        fused = self.norm(seq_feat + region_feat + macro_feat)
        logits = self.classifier(fused)
        return logits
