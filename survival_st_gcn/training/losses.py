"""训练用损失函数

包含：
- BinaryFocalLoss：适合类别不平衡的二分类任务，强调难分类样本；
  建议与 logits 一起使用（内部做 sigmoid）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BinaryFocalLoss(nn.Module):
    """二分类 Focal Loss（适配不平衡数据集）。"""

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, size_average: bool = True) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 先将 logits 转换为概率，用于 focal loss 计算
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        log_p_t = torch.log(p_t)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = -alpha_t * torch.pow((1 - p_t), self.gamma) * log_p_t
        return loss.mean() if self.size_average else loss.sum()
