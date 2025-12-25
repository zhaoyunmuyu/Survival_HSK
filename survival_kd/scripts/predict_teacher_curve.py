"""使用教师模型按年份预测单个餐厅的“仍营业概率”并绘制折线图。

用法示例：
  python -m survival_kd.scripts.predict_teacher_curve ^
      --restaurant-id 4221 ^
      --checkpoint checkpoints_kd/teacher_best.pt ^
      --hidden-dim 512 ^
      --min-year 2010 --max-year 2019 --time-shift-years 0

说明：
- 教师模型（MambaTeacher）默认使用截至 reference_year 的全部历史评论（避免“未来评论泄露”）；
- 输出为 sigmoid(logits) 的概率，表示 is_open=1（仍营业）的概率；
- 需要安装 matplotlib 才能绘图。
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
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

from survival_st_gcn.utils.constants import REGION_MAPPING
from survival_st_gcn.utils.paths import resolve_data_file, resolve_data_dir
from survival_kd.data.reviews import build_restaurant_review_cache, prepare_review_dataframe
from survival_kd.models.mamba_teacher import MambaTeacher
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
    *,
    history_mode: str = "all",
    history_years: int = 2,
) -> Dict[str, torch.Tensor]:
    text = reviews["text"]
    images = reviews["images"]
    features = reviews["features"]
    years_tensor = reviews["years"].long()

    if history_mode == "last2":
        hist = (years_tensor <= year) & (years_tensor >= (year - 1)) & (years_tensor > 0)
    elif history_mode == "all":
        hist = (years_tensor <= year) & (years_tensor > 0)
    elif history_mode == "years":
        win = max(1, int(history_years))
        hist = (years_tensor <= year) & (years_tensor >= (year - (win - 1))) & (years_tensor > 0)
    else:
        raise ValueError(f"Unknown history_mode: {history_mode!r}")

    # 通过置零让 padding_mask 生效，避免未来评论泄露
    hist_f = hist.to(dtype=torch.float32)
    text = text * hist_f.unsqueeze(-1)
    images = images * hist_f.unsqueeze(-1)
    features = features * hist_f.unsqueeze(-1)
    masked_years = torch.where(hist, years_tensor, torch.zeros_like(years_tensor))

    region_macro = macro_data.get(region_key, {}) if region_key else {}
    macro_feat = region_macro.get(year, macro_default)
    region_encoding = REGION_MAPPING.get(region_key, 0) if region_key else 0

    batch: Dict[str, torch.Tensor] = {
        "review_text": text.unsqueeze(0),
        "review_images": images.unsqueeze(0),
        "review_features": features.unsqueeze(0),
        "review_years": masked_years.unsqueeze(0),
        "last2_mask": hist.unsqueeze(0),  # 仅用于 debug/计数；teacher 不依赖该字段
        "reference_year": torch.tensor([int(year)], dtype=torch.long),
        "macro_features": macro_feat.unsqueeze(0),
        "region_encoding": torch.tensor([float(region_encoding)], dtype=torch.float32),
    }
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict yearly open probability for one restaurant using teacher model.")
    parser.add_argument("--restaurant-id", type=str, required=True, help="餐厅的 restaurant_id（与 parquet 中一致）")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_kd/teacher_best.pt", help="教师模型权重路径")
    parser.add_argument("--hidden-dim", type=int, default=512, help="教师模型 d_model（需与训练时一致）")
    parser.add_argument("--min-year", type=int, default=None, help="起始评论年份（默认自动取该店最早评论年份）")
    parser.add_argument("--max-year", type=int, default=None, help="结束评论年份（默认自动取该店最晚评论年份）")
    parser.add_argument("--output", type=str, default="", help="折线图输出路径（默认 teacher_curve_<id>.png）")
    parser.add_argument("--time-shift-years", type=int, default=0, help="横坐标年份平移（默认 0）")
    parser.add_argument(
        "--history",
        type=str,
        choices=["all", "last2", "years"],
        default="all",
        help="历史评论范围：all=截至该年的全部历史；last2=仅当年+前一年；years=最近N年",
    )
    parser.add_argument("--history-years", type=int, default=2, help="当 --history=years 时生效：最近 N 年（含当年）")
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
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.checkpoint}")

    teacher = MambaTeacher(d_model=args.hidden_dim)
    state = torch.load(args.checkpoint, map_location="cpu")
    teacher.load_state_dict(state.get("state", state))
    teacher.to(device)
    teacher.eval()

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
    used_counts: List[int] = []

    with torch.no_grad():
        for year in range(min_year, max_year + 1):
            batch = _build_batch_for_year(
                year,
                reviews,
                macro_data,
                macro_default,
                region_key,
                history_mode=args.history,
                history_years=args.history_years,
            )
            used_counts.append(int(batch["last2_mask"].sum().item()))
            for key, value in batch.items():
                batch[key] = value.to(device)
            logits = teacher(batch)
            prob_open = torch.sigmoid(logits).detach().cpu().item()
            years.append(year)
            probs.append(float(prob_open))

    print(f"Restaurant id: {args.restaurant_id}")
    name = row.get("name") or row.get("restaurant_name") or ""
    if name:
        print(f"Name: {name}")
    print("Yearly open probabilities (teacher model, by reference year):")
    for y, p, n_used in zip(years, probs, used_counts):
        yy = y + args.time_shift_years
        print(f"- reference year {y} -> x {yy}: {p:.4f} | used_reviews={n_used}")

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

    output_path = args.output or f"teacher_curve_{args.restaurant_id}.png"
    plt.figure(figsize=(6, 4))
    plt.plot(x_plot, y_plot, marker="o")
    plt.xlabel("Year" if not args.time_shift_years else f"Year (reference year + {args.time_shift_years})")
    plt.ylabel("Open probability (is_open)")
    plt.title(f"Teacher model open prob - {args.restaurant_id}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved line chart to: {output_path}")


if __name__ == "__main__":
    main()

