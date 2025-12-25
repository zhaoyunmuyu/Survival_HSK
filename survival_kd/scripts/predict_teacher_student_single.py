"""用与训练一致的数据构造方式，构建单个餐厅样本并分别用 Teacher/Student 推理。

示例：
  python -m survival_kd.scripts.predict_teacher_student_single ^
      --restaurant-id 4221 ^
      --teacher-checkpoint checkpoints_kd/teacher_best.pt ^
      --student-checkpoint checkpoints_kd/student_best.pt ^
      --hidden-dim 512 ^
      --reference-year 2019

输出：
- Teacher/Student 的 sigmoid(logits) 概率（当前项目口径：is_open=1 为“仍营业”）。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch

from survival_st_gcn.utils.constants import REGION_MAPPING
from survival_st_gcn.utils.paths import resolve_data_dir, resolve_data_file
from survival_kd.data.reviews import build_restaurant_review_cache, prepare_review_dataframe
from survival_kd.models.bilstm_student import BiLSTMStudent
from survival_kd.models.mamba_teacher import MambaTeacher
from survival_st_gcn.data.macro import prepare_macro_data
from survival_st_gcn.data.text_vectors import build_text_vector_map


def _resolve_data_file_local(filename: str) -> str:
    path = resolve_data_file(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {filename} (resolved to {path})")
    return path


def _read_parquet_df(path: str):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pandas()


def _load_single_restaurant_row(restaurant_id: str) -> Dict:
    rest_df = _read_parquet_df(_resolve_data_file_local("restaurant_data.parquet"))
    rest_df["restaurant_id"] = rest_df["restaurant_id"].astype(str)
    sub = rest_df[rest_df["restaurant_id"] == str(restaurant_id)]
    if sub.empty:
        raise KeyError(f"Restaurant id {restaurant_id!r} not found in restaurant_data.parquet")
    return sub.iloc[0].to_dict()


def _load_single_restaurant_reviews(restaurant_id: str):
    review_df = _read_parquet_df(_resolve_data_file_local("review_data.parquet"))
    review_df["restaurant_id"] = review_df["restaurant_id"].astype(str)
    review_df = review_df[review_df["restaurant_id"] == str(restaurant_id)]
    if review_df.empty:
        raise RuntimeError(f"No reviews found for restaurant id {restaurant_id!r}")
    return prepare_review_dataframe(review_df)


def _history_mask(years_tensor: torch.Tensor, reference_year: int, history: str, history_years: int) -> torch.Tensor:
    years_tensor = years_tensor.long()
    if history == "last2":
        return (years_tensor <= reference_year) & (years_tensor >= (reference_year - 1)) & (years_tensor > 0)
    if history == "all":
        return (years_tensor <= reference_year) & (years_tensor > 0)
    if history == "years":
        win = max(1, int(history_years))
        return (years_tensor <= reference_year) & (years_tensor >= (reference_year - (win - 1))) & (years_tensor > 0)
    raise ValueError(f"Unknown history: {history!r}")


def _build_single_batch(
    *,
    restaurant_id: str,
    reference_year: int,
    history: str,
    history_years: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    row = _load_single_restaurant_row(restaurant_id)
    review_df = _load_single_restaurant_reviews(restaurant_id)

    text_feat_df = _read_parquet_df(_resolve_data_file_local("text_vectors.parquet"))
    if "text_vector" not in text_feat_df.columns and "element" in text_feat_df.columns:
        text_feat_df = text_feat_df.rename(columns={"element": "text_vector"})
    text_vector_map = build_text_vector_map(text_feat_df)

    img_dirs = tuple(
        resolve_data_dir(d)
        for d in (
            "precomputed_img_features",
            "precomputed_img_features1",
            "precomputed_img_features2",
        )
    )
    review_cache = build_restaurant_review_cache(
        review_df,
        text_vector_map,
        img_feat_dirs=img_dirs,
    )
    if str(restaurant_id) not in review_cache:
        raise RuntimeError(f"Failed to build review cache for restaurant id {restaurant_id!r}")
    reviews = review_cache[str(restaurant_id)]

    with open(_resolve_data_file_local("normalized_macro_data.json"), "r", encoding="utf-8") as handle:
        macro_raw = json.load(handle)
    macro_data, macro_default = prepare_macro_data(macro_raw)

    region_key = row.get("region_code") if isinstance(row.get("region_code"), str) else None
    region_macro = macro_data.get(region_key, {}) if region_key else {}
    macro_feat = region_macro.get(int(reference_year), macro_default)
    macro_hit = bool(region_key) and (int(reference_year) in region_macro)

    years_tensor = reviews["years"].long()
    hist_mask = _history_mask(years_tensor, int(reference_year), history, history_years)
    region_encoding = REGION_MAPPING.get(region_key, 0) if region_key else 0

    is_open_value = float(row.get("is_open") or 0.0)

    batch: Dict[str, torch.Tensor] = {
        "restaurant_id": torch.tensor([int(restaurant_id)], dtype=torch.long),
        "is_open": torch.tensor([is_open_value], dtype=torch.float32),
        "review_text": reviews["text"].unsqueeze(0),
        "review_images": reviews["images"].unsqueeze(0),
        "review_features": reviews["features"].unsqueeze(0),
        "review_years": years_tensor.unsqueeze(0),
        "last2_mask": hist_mask.unsqueeze(0),
        "macro_features": macro_feat.unsqueeze(0),
        "region_encoding": torch.tensor([float(region_encoding)], dtype=torch.float32),
        "reference_year": torch.tensor([int(reference_year)], dtype=torch.long),
    }

    meta: Dict[str, object] = {
        "restaurant_id": str(restaurant_id),
        "is_open": int(is_open_value),
        "region_code": region_key,
        "operation_latest_year": row.get("operation_latest_year"),
        "reference_year": int(reference_year),
        "history": history,
        "history_years": int(history_years),
        "n_reviews_total": int((years_tensor > 0).sum().item()),
        "n_history_reviews": int(hist_mask.sum().item()),
        "macro_hit": bool(macro_hit),
    }
    return batch, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one restaurant batch (training-style) and predict with teacher & student.")
    parser.add_argument("--restaurant-id", type=str, required=True)
    parser.add_argument("--teacher-checkpoint", type=str, default="checkpoints_kd/teacher_best.pt")
    parser.add_argument("--student-checkpoint", type=str, default="checkpoints_kd/student_best.pt")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--reference-year", type=int, default=2019, help="与训练口径对齐时通常为 2019")
    parser.add_argument("--history", type=str, choices=["all", "last2", "years"], default="last2", help="构造 last2_mask 的口径")
    parser.add_argument("--history-years", type=int, default=2, help="当 --history=years 时生效")
    parser.add_argument("--device", type=str, default="", help="可选：cpu/cuda；默认自动选择")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    if not os.path.exists(args.teacher_checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")
    if not os.path.exists(args.student_checkpoint):
        raise FileNotFoundError(f"Student checkpoint not found: {args.student_checkpoint}")

    batch, meta = _build_single_batch(
        restaurant_id=args.restaurant_id,
        reference_year=args.reference_year,
        history=args.history,
        history_years=args.history_years,
    )
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    teacher = MambaTeacher(d_model=args.hidden_dim).to(device)
    teacher_state = torch.load(args.teacher_checkpoint, map_location="cpu")
    teacher.load_state_dict(teacher_state.get("state", teacher_state))
    teacher.eval()

    student = BiLSTMStudent(d_model=args.hidden_dim).to(device)
    student_state = torch.load(args.student_checkpoint, map_location="cpu")
    student.load_state_dict(student_state.get("state", student_state))
    student.eval()

    with torch.no_grad():
        t_prob = float(torch.sigmoid(teacher(batch)).detach().cpu().item())
        s_prob = float(torch.sigmoid(student(batch)).detach().cpu().item())

    print(json.dumps(meta, ensure_ascii=False))
    print(f"teacher_open_prob={t_prob:.6f}")
    print(f"student_open_prob={s_prob:.6f}")
    # label check
    if isinstance(meta.get("is_open"), int):
        print(f"true_is_open={meta['is_open']}")


if __name__ == "__main__":
    main()

