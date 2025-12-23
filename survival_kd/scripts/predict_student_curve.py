"""使用学生模型按年份预测单个餐厅的死亡概率并绘制折线图。

用法示例：
  python -m survival_kd.scripts.predict_student_curve ^
      --restaurant-id 123456 ^
      --checkpoint checkpoints_kd/student_best.pt ^
      --hidden-dim 512

说明：
- 依赖训练好的学生模型（BiLSTMStudent），默认从 checkpoints_kd/student_best.pt 加载；
- 自动按该餐厅评论出现的年份范围逐年预测死亡概率（1 - is_open），并保存折线图 PNG。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - 图形依赖可选
    plt = None  # type: ignore[assignment]

from survival_st_gcn.utils.constants import REGION_MAPPING
from survival_st_gcn.utils.paths import resolve_data_file, resolve_data_dir
from survival_kd.data.reviews import build_restaurant_review_cache, prepare_review_dataframe
from survival_kd.models.bilstm_student import BiLSTMStudent
from survival_kd.calibration import LogitCalibrator
from survival_st_gcn.data.text_vectors import build_text_vector_map
from survival_st_gcn.data.macro import prepare_macro_data


def _resolve_data_file_local(filename: str) -> str:
    path = resolve_data_file(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {filename} (resolved to {path})")
    return path


def _read_parquet_df(path: str):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pandas()


def _load_single_restaurant_context(
    restaurant_id: str,
) -> Tuple[Dict, Dict[str, torch.Tensor], Dict[str, Dict[int, torch.Tensor]], torch.Tensor, str | None]:
    rest_df = _read_parquet_df(_resolve_data_file_local("restaurant_data.parquet"))
    rest_df["restaurant_id"] = rest_df["restaurant_id"].astype(str)
    row_df = rest_df[rest_df["restaurant_id"] == restaurant_id]
    if row_df.empty:
        raise KeyError(f"Restaurant id {restaurant_id!r} not found in restaurant_data.parquet")
    row = row_df.iloc[0].to_dict()

    review_df = _read_parquet_df(_resolve_data_file_local("review_data.parquet"))
    review_df["restaurant_id"] = review_df["restaurant_id"].astype(str)
    review_df = review_df[review_df["restaurant_id"] == restaurant_id]
    if review_df.empty:
        raise RuntimeError(f"No reviews found for restaurant id {restaurant_id!r}")
    review_df = prepare_review_dataframe(review_df)

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
    if restaurant_id not in review_cache:
        raise RuntimeError(f"Failed to build review cache for restaurant id {restaurant_id!r}")
    reviews = review_cache[restaurant_id]

    with open(_resolve_data_file_local("normalized_macro_data.json"), "r", encoding="utf-8") as handle:
        macro_raw = json.load(handle)
    macro_data, macro_default = prepare_macro_data(macro_raw)

    region_key = row.get("region_code") if isinstance(row.get("region_code"), str) else None
    return row, reviews, macro_data, macro_default, region_key


def _build_batch_for_year(
    year: int,
    reviews: Dict[str, torch.Tensor],
    macro_data: Dict[str, Dict[int, torch.Tensor]],
    macro_default: torch.Tensor,
    region_key: str | None,
) -> Dict[str, torch.Tensor]:
    text = reviews["text"]
    images = reviews["images"]
    features = reviews["features"]
    years_tensor = reviews["years"].long()

    # 防止“未来评论泄露”：只允许使用 <= 当前评估 year 的历史评论
    last2 = (years_tensor <= year) & (years_tensor >= (year - 1)) & (years_tensor > 0)

    region_macro = macro_data.get(region_key, {}) if region_key else {}
    macro_feat = region_macro.get(year, macro_default)

    region_encoding = REGION_MAPPING.get(region_key, 0) if region_key else 0

    batch: Dict[str, torch.Tensor] = {
        "review_text": text.unsqueeze(0),
        "review_images": images.unsqueeze(0),
        "review_features": features.unsqueeze(0),
        "review_years": years_tensor.unsqueeze(0),
        "last2_mask": last2.unsqueeze(0),
        "reference_year": torch.tensor([int(year)], dtype=torch.long),
        "macro_features": macro_feat.unsqueeze(0),
        "region_encoding": torch.tensor([float(region_encoding)], dtype=torch.float32),
    }
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict yearly death probability for one restaurant using student model."
    )
    parser.add_argument("--restaurant-id", type=str, required=True, help="餐厅的 restaurant_id（与 parquet 中一致）")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_kd/student_best.pt", help="学生模型权重路径")
    parser.add_argument(
        "--calibrator-checkpoint",
        type=str,
        default="",
        help="可选：logit 标定层 checkpoint 路径（用 train_calibrator 训练获得）",
    )
    parser.add_argument("--hidden-dim", type=int, default=512, help="学生模型 d_model（需与训练时一致）")
    parser.add_argument("--min-year", type=int, default=None, help="起始评论年份（默认自动取该店最早评论年份）")
    parser.add_argument("--max-year", type=int, default=None, help="结束评论年份（默认自动取该店最晚评论年份）")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="折线图输出路径（默认 student_curve_<id>.png）",
    )
    parser.add_argument(
        "--time-shift-years",
        type=int,
        default=5,
        help="横坐标年份相对于评论年份的平移（默认 +5，表示用五年前评论预测当前状态）",
    )
    parser.add_argument(
        "--plot-time-step",
        type=str,
        choices=["year", "half-year", "quarter"],
        default="year",
        help="绘图时横坐标步长，仅影响插值后的曲线，不改变模型按年预测的结果",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Student checkpoint not found: {args.checkpoint}")

    student = BiLSTMStudent(d_model=args.hidden_dim)
    state = torch.load(args.checkpoint, map_location="cpu")
    student.load_state_dict(state.get("state", state))
    student.to(device)
    student.eval()

    calibrator: LogitCalibrator | None = None
    if args.calibrator_checkpoint:
        if not os.path.exists(args.calibrator_checkpoint):
            raise FileNotFoundError(f"Calibrator checkpoint not found: {args.calibrator_checkpoint}")
        cal_state = torch.load(args.calibrator_checkpoint, map_location="cpu")
        calibrator = LogitCalibrator()
        calibrator.load_state_dict(cal_state.get("state", cal_state))
        calibrator.to(device)
        calibrator.eval()

    row, reviews, macro_data, macro_default, region_key = _load_single_restaurant_context(args.restaurant_id)
    years_tensor = reviews["years"].long()
    year_values = sorted({int(y) for y in years_tensor.view(-1).tolist() if int(y) > 0})
    if not year_values:
        raise RuntimeError(f"No valid review years found for restaurant id {args.restaurant_id!r}")

    min_year = args.min_year if args.min_year is not None else year_values[0]
    max_year = args.max_year if args.max_year is not None else year_values[-1]
    if min_year > max_year:
        raise ValueError(f"min-year ({min_year}) should not be greater than max-year ({max_year})")

    years: List[int] = []
    probs: List[float] = []

    with torch.no_grad():
        for year in range(min_year, max_year + 1):
            batch = _build_batch_for_year(year, reviews, macro_data, macro_default, region_key)
            for key, value in batch.items():
                batch[key] = value.to(device)
            logits = student(batch)
            if calibrator is not None:
                logits = calibrator(logits)
            prob = torch.sigmoid(logits).detach().cpu().item()
            years.append(year)
            probs.append(float(prob))

    print(f"Restaurant id: {args.restaurant_id}")
    name = row.get("name") or row.get("restaurant_name") or ""
    if name:
        print(f"Name: {name}")
    print("Yearly death probabilities (student model, by review year):")
    for review_year, prob in zip(years, probs):
        target_year = review_year + args.time_shift_years
        print(f"- review year {review_year} -> target year {target_year}: {prob:.4f}")

    if plt is None:
        print("matplotlib is not installed; skip plotting. Install matplotlib to get line charts.")
        return

    eval_years = np.asarray(years, dtype=float) + float(args.time_shift_years)
    x_plot = eval_years
    y_plot = np.asarray(probs, dtype=float)
    if args.plot_time_step != "year" and len(eval_years) > 1:
        step = 0.5 if args.plot_time_step == "half-year" else 0.25
        x_new = np.arange(eval_years[0], eval_years[-1] + 1e-8, step)
        y_new = np.interp(x_new, eval_years, y_plot)
        x_plot, y_plot = x_new, y_new

    output_path = args.output or f"student_curve_{args.restaurant_id}.png"
    plt.figure(figsize=(6, 4))
    plt.plot(x_plot, y_plot, marker="o")
    xlabel = "Year"
    if args.time_shift_years:
        xlabel = f"Year (review year + {args.time_shift_years})"
    plt.xlabel(xlabel)
    plt.ylabel("Death probability (1 - is_open)")
    plt.title(f"Student model death prob - {args.restaurant_id}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved line chart to: {output_path}")


if __name__ == "__main__":
    main()
