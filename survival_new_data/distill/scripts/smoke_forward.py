"""最小前向自检（不依赖真实数据）。

运行：
  python -m survival_new_data.distill.scripts.smoke_forward
"""

from __future__ import annotations

import torch

from survival_new_data.distill.models import BiLSTMStudent, MambaTeacher


def main() -> None:
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 6
    text_dim = 8
    img_dim = 4
    feature_dim = 3
    hidden_dim = 16

    review_text = torch.randn(batch_size, seq_len, text_dim)
    review_images = torch.randn(batch_size, seq_len, img_dim)
    review_features = torch.randn(batch_size, seq_len, feature_dim)

    # padding in tail
    review_text[:, -2:] = 0
    review_images[:, -2:] = 0
    review_features[:, -2:] = 0

    review_time = torch.tensor([[2020, 2021, 2022, 2023, 0, 0], [2019, 2020, 2021, 0, 0, 0]], dtype=torch.long)
    last2_mask = torch.tensor([[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0]], dtype=torch.bool)
    reference_time = torch.tensor([2023, 2021], dtype=torch.long)

    batch = {
        "review_text": review_text,
        "review_images": review_images,
        "review_features": review_features,
        "review_time": review_time,
        "last2_mask": last2_mask,
        "reference_time": reference_time,
        "region_encoding": torch.tensor([[1.0], [3.0]], dtype=torch.float32),
    }

    teacher = MambaTeacher(
        d_model=hidden_dim,
        text_dim=text_dim,
        img_dim=img_dim,
        feature_dim=feature_dim,
        time_step="year",
        max_offset_years=5,
    )
    student = BiLSTMStudent(
        d_model=hidden_dim,
        text_dim=text_dim,
        img_dim=img_dim,
        feature_dim=feature_dim,
        time_step="year",
        max_offset_years=5,
    )

    with torch.no_grad():
        logits_teacher = teacher(batch)
        logits_student = student(batch)

    print("OK")
    print("- teacher logits:", tuple(logits_teacher.shape))
    print("- student logits:", tuple(logits_student.shape))


if __name__ == "__main__":
    main()
