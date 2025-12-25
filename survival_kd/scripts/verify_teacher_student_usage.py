"""
验证教师/学生使用的序列范围差异，并做一次最小训练步：

- 教师：使用所有非 padding 的时间步；
- 学生：仅使用“最后两年”的时间步（其余视为 padding 屏蔽）；

运行：  python -m survival_kd.scripts.verify_teacher_student_usage
"""

from __future__ import annotations

import os
from typing import Dict

import torch

from survival_kd.data.loaders import prepare_dataloaders_kd
from survival_kd.models.mamba_teacher import MambaTeacher
from survival_kd.models.bilstm_student import BiLSTMStudent
from survival_st_gcn.training.losses import BinaryFocalLoss


def _padding_mask_from_batch(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """与 TokenBuilder 内部一致：文本/图片均为全零处视作 padding。"""
    text = batch["review_text"]  # [B,L,300]
    images = batch["review_images"]  # [B,L,512]
    text_zero = (text.abs().sum(dim=-1) == 0)
    img_zero = (images.abs().sum(dim=-1) == 0)
    return text_zero & img_zero  # [B,L]


def main() -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    device = torch.device("cpu")

    loaders = prepare_dataloaders_kd(batch_size=8, limit_restaurants=None)
    train_loader = loaders["train_loader"]

    # 取一个 batch 进行掩码统计
    batch = next(iter(train_loader))
    padding_mask = _padding_mask_from_batch(batch)  # [B,L]
    last2_mask = batch["last2_mask"].to(torch.bool)

    teacher_used = (~padding_mask).sum(dim=1)  # 每个样本，教师有效 token 数
    student_used = ((~padding_mask) & last2_mask).sum(dim=1)  # 学生有效 token 数（仅最后两年）

    print("样本级 token 使用统计（前 8 条）：")
    for i in range(min(teacher_used.numel(), 8)):
        print(f"- sample[{i}]: teacher_tokens={int(teacher_used[i])} student_tokens={int(student_used[i])}")

    # 最小化训练各一步
    teacher = MambaTeacher(d_model=512).to(device)
    student = BiLSTMStudent(d_model=512).to(device)
    criterion = BinaryFocalLoss(alpha=0.7, gamma=3)

    # 教师监督一步
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    labels = batch["is_open"].float().view(-1, 1)

    teacher.train()
    t_out = teacher(batch)
    t_loss = criterion(t_out, labels * 0.8 + 0.1)
    t_loss.backward()
    print(f"教师一步训练完成：loss={float(t_loss.item()):.4f}")

    # 学生蒸馏一步（使用与教师相同 batch，但内部只用最后两年）
    student.train()
    with torch.no_grad():
        t_logits = teacher(batch)
    s_logits = student(batch)
    # KL 蒸馏（概率 + 温度）
    T = 2.0
    p_t = torch.sigmoid(t_logits / T)
    p_s = torch.sigmoid(s_logits / T)
    eps = 1e-7
    kl = (p_t.clamp(eps, 1 - eps) * torch.log(p_t.clamp(eps, 1 - eps) / p_s.clamp(eps, 1 - eps))
          + (1 - p_t).clamp(eps, 1 - eps) * torch.log((1 - p_t).clamp(eps, 1 - eps) / (1 - p_s).clamp(eps, 1 - eps))).mean()
    sup = criterion(s_logits, labels * 0.8 + 0.1)
    s_loss = sup + (T * T) * 0.7 * kl
    s_loss.backward()
    print(f"学生一步训练完成：sup={float(sup.item()):.4f} kd={float(((T*T)*0.7*kl).item()):.4f} total={float(s_loss.item()):.4f}")


if __name__ == "__main__":
    main()
