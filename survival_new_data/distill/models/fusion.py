"""模态融合与 Token 构建（HSK 蒸馏迁移版）

功能：
- 将文本/图片/结构化评论特征对齐到同一维度并融合为 token；
- 依据参考年与评论年份加入“时序偏移嵌入”，显式编码“新近性”；
- 输出 token 序列与 padding 掩码，供教师/学生模型消费。

与 survival_kd 版本的主要差异：
- 不再依赖全局常量维度（TEXT_VECTOR_DIM=300 等），而改为从构造参数传入，
  以便直接对接 survival_hsk 的 BERT 768 维向量。
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn


class TokenBuilder(nn.Module):
    """将评论多路特征融合为定长 token，并叠加时间嵌入。"""

    _TimeStep = Literal["year", "half", "quarter"]

    def __init__(
        self,
        *,
        d_model: int = 512,
        text_dim: int = 768,
        img_dim: int = 512,
        feature_dim: int = 8,
        time_step: _TimeStep = "year",
        max_offset_years: int = 10,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.time_step: TokenBuilder._TimeStep = time_step
        periods_per_year = {"year": 1, "half": 2, "quarter": 4}[self.time_step]
        self.max_time_offset = int(max_offset_years) * periods_per_year

        self.text_proj = nn.Linear(int(text_dim), self.d_model)
        self.img_proj = nn.Linear(int(img_dim), self.d_model)
        self.struct_proj = nn.Linear(int(feature_dim), self.d_model)

        # 门控融合（逐维 gate）
        self.gate = nn.Linear(self.d_model * 3, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

        # 时间偏移嵌入：
        # - 0 代表“未知”（review_time 或 reference_time 缺失）
        # - 1..max_time_offset+1 代表 offset=0..max_time_offset
        self.time_emb = nn.Embedding(self.max_time_offset + 2, self.d_model)

    @staticmethod
    def _build_padding_mask(text: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """基于文本/图片全零判断 padding 位置（True=padding）。"""
        text_zero = text.abs().sum(dim=-1) == 0
        img_zero = images.abs().sum(dim=-1) == 0
        return text_zero & img_zero

    def _time_offset_idx(self, review_time: torch.Tensor, reference_time: torch.Tensor) -> torch.Tensor:
        """将时间差转换为离散索引（单位由 time_step 决定）。

        - 若 review_time<=0 或 reference_time<=0，返回 0（未知）。
        - 否则 offset = clamp(ref - review, 0, max_time_offset)，并映射到 [1..max_time_offset+1]。
        """
        ref = reference_time.view(-1, 1).to(review_time.dtype)
        t = review_time.to(review_time.dtype)
        unknown = (t <= 0) | (ref <= 0)
        raw = torch.clamp(ref - t, min=0, max=self.max_time_offset)
        idx = (raw.long() + 1)
        return torch.where(unknown, torch.zeros_like(idx), idx)

    def forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """输入 batch 字典，输出 tokens[B,L,d] 与 padding_mask[B,L]。

        期望字段（dtype/shape 由上游保证）：
        - review_text: [B, L, text_dim]
        - review_images: [B, L, img_dim]
        - review_features: [B, L, feature_dim]
        - 时间索引（推荐字段名）：
          - review_time: [B, L]
          - reference_time: [B]
        - 为兼容旧命名，也接受：
          - review_years: [B, L]
          - reference_year: [B]
        """
        text = batch["review_text"]
        images = batch["review_images"]
        struct = batch["review_features"]
        review_time = batch.get("review_time")
        if review_time is None:
            review_time = batch["review_years"]
        reference_time = batch.get("reference_time")
        if reference_time is None:
            reference_time = batch["reference_year"]
        reference_time = reference_time.view(-1)

        text_d = self.text_proj(text)
        img_d = self.img_proj(images)
        struct_d = self.struct_proj(struct)

        gate_inp = torch.cat([text_d, img_d, struct_d], dim=-1)
        gate = torch.sigmoid(self.gate(gate_inp))
        fused = gate * text_d + (1.0 - gate) * img_d + struct_d

        offset_idx = self._time_offset_idx(review_time, reference_time)
        offset_embed = self.time_emb(offset_idx)
        tokens = self.norm(fused + offset_embed)

        padding_mask = self._build_padding_mask(text, images)
        return tokens, padding_mask
