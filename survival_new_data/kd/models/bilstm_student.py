"""学生模型：BiLSTM 序列编码 + 静态特征融合（仅“最后两年”）

说明：
- 使用与教师相同的 TokenBuilder 保持输入对齐；
- 根据 batch["last2_mask"] 屏蔽非最后两年的 token；
- 双向 LSTM 编码后做掩码均值池化，再与静态特征融合分类。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from survival_new_data.kd.models.fusion import TokenBuilder


class BiLSTMStudent(nn.Module):
    """学生模型：双向 LSTM 序列编码。"""

    def __init__(self, d_model: int = 512, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_builder = TokenBuilder(d_model=d_model)

        hidden = d_model // 2
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.region_proj = nn.Linear(18, d_model)
        self.macro_proj = nn.Sequential(
            nn.Linear(62, d_model),
            nn.ReLU(),
        )

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

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (~mask).float()
        denom = valid.sum(dim=1).clamp(min=1.0)  # [B]
        summed = (x * valid.unsqueeze(-1)).sum(dim=1)  # [B, d]
        return summed / denom.unsqueeze(-1)

    def _region_feature(self, region_encoding: torch.Tensor) -> torch.Tensor:
        region_idx = region_encoding.squeeze().long() - 1
        region_onehot = F.one_hot(region_idx.clamp(min=0), num_classes=18).float()
        return self.region_proj(region_onehot)

    def forward(self, batch: dict) -> torch.Tensor:
        tokens, padding_mask = self.token_builder(batch)  # [B,L,d], [B,L]

        # 仅保留最后两年：将非 last2 的位置视作 padding
        last2_mask = batch["last2_mask"].to(torch.bool)  # True 表示“属于最后两年”
        keep_mask = last2_mask  # [B,L]
        # 将非 keep 的位标记为 padding（True），以共同参与掩码
        effective_padding = padding_mask | (~keep_mask)

        lstm_out, _ = self.lstm(tokens)
        seq_feat = self._masked_mean(lstm_out, effective_padding)

        region_feat = self._region_feature(batch["region_encoding"])  # [B,d]
        macro_feat = self.macro_proj(batch["macro_features"])  # [B,62]->[B,d]
        fused = self.norm(seq_feat + region_feat + macro_feat)
        logits = self.classifier(fused)
        return logits
