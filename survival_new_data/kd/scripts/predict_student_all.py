"""批量使用学生模型为所有餐厅按年份预测死亡概率，并导出 CSV。

用法示例：
  python -m survival_hsk.kd.scripts.predict_student_all ^
      --checkpoint checkpoints_kd/student_best.pt ^
      --hidden-dim 512 ^
      --output student_all_yearly_probs.csv

说明：
- 对 restaurant_data.parquet 中的每一家餐厅，结合其所有评论与宏观数据；
- 对该餐厅所有评论年份（可用 --min-year/--max-year 限制）逐年构造样本，
  使用学生模型预测“死亡概率”（训练时目标为 1 - is_open）；
- 结果保存为 CSV：列包括 restaurant_id, year, pred_death_prob 等。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - 图形依赖可选
    plt = None  # type: ignore[assignment]

from survival.utils.constants import REGION_MAPPING
from survival.utils.paths import resolve_data_file, resolve_data_dir
from survival_new_data.kd.data.reviews import build_restaurant_review_cache, prepare_review_dataframe
from survival_new_data.kd.models.bilstm_student import BiLSTMStudent
from survival_new_data.kd.calibration import LogitCalibrator
from survival.data.text_vectors import build_text_vector_map
from survival.data.macro import prepare_macro_data


def _resolve_data_file_local(filename: str) -> str:
    path = resolve_data_file(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {filename} (resolved to {path})")
    return path


def _read_parquet_df(path: str):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pandas()


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
    parser = argparse.ArgumentParser(description="Predict yearly death probability for all restaurants using student model.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_kd/student_best.pt", help="学生模型权重路径")
    parser.add_argument(
        "--calibrator-checkpoint",
        type=str,
        default="",
        help="可选：logit 标定层 checkpoint 路径（用 train_calibrator 训练获得）",
    )
    parser.add_argument("--hidden-dim", type=int, default=512, help="学生模型 d_model（需与训练时一致）")
    parser.add_argument("--min-year", type=int, default=None, help="全局起始评论年份（默认按各店铺最早评论年份）")
    parser.add_argument("--max-year", type=int, default=None, help="全局结束评论年份（默认按各店铺最晚评论年份）")
    parser.add_argument("--output", type=str, default="student_all_yearly_probs.csv", help="输出 CSV 文件路径")
    parser.add_argument(
        "--max-restaurants",
        type=int,
        default=None,
        help="最多预测的餐厅数量（按 restaurant_data.parquet 中顺序截断）",
    )
    parser.add_argument(
        "--plot-restaurant-id",
        type=str,
        default="",
        help="若非空，则对该餐厅按年份绘制学生模型死亡概率折线图，并保存 PNG 文件",
    )
    parser.add_argument(
        "--plot-output-dir",
        type=str,
        default="",
        help="图像输出目录（默认与 CSV 同一目录）",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="为所有已预测的餐厅按年份批量绘制折线图",
    )
    parser.add_argument(
        "--plot-time-step",
        type=str,
        choices=["year", "half-year", "quarter"],
        default="year",
        help="绘图时横坐标步长，仅影响插值后的曲线，不改变模型按年预测的结果",
    )
    parser.add_argument(
        "--csv-time-step",
        type=str,
        choices=["year", "half-year", "quarter"],
        default="year",
        help="输出 CSV 时的时间粒度：按年 / 半年 / 季度（半年/季度通过在 year 上线性插值获得）",
    )
    return parser.parse_args()


def _plot_restaurant_curve(
    df: pd.DataFrame,
    restaurant_id: str,
    base_dir: str,
    plot_time_step: str = "year",
) -> None:
    sub_df = df[df["restaurant_id"] == restaurant_id].copy()
    if sub_df.empty:
        print(f"No prediction rows found for restaurant_id={restaurant_id!r}; skip plotting.")
        return

    sub_df = sub_df.sort_values("year")
    years = sub_df["year"].astype(float).to_numpy()
    probs = sub_df["pred_death_prob"].astype(float).to_numpy()
    name = sub_df["name"].iloc[0] if "name" in sub_df.columns else ""

    x_plot = years
    y_plot = probs
    if plot_time_step != "year" and len(x_plot) > 1:
        step = 0.5 if plot_time_step == "half-year" else 0.25
        x_new = np.arange(x_plot[0], x_plot[-1] + 1e-8, step)
        y_new = np.interp(x_new, x_plot, y_plot)
        x_plot, y_plot = x_new, y_new

    os.makedirs(base_dir, exist_ok=True)
    fig_path = os.path.join(base_dir, f"student_curve_{restaurant_id}.png")

    plt.figure(figsize=(6, 4))
    plt.plot(x_plot, y_plot, marker="o")
    plt.xlabel("Year")
    plt.ylabel("Death probability (1 - is_open)")
    title = f"Student death prob - {restaurant_id}"
    if isinstance(name, str) and name:
        title += f" ({name})"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Saved line chart for restaurant_id={restaurant_id} to: {fig_path}")


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

    rest_df = _read_parquet_df(_resolve_data_file_local("restaurant_data.parquet"))
    rest_df["restaurant_id"] = rest_df["restaurant_id"].astype(str)

    review_df = _read_parquet_df(_resolve_data_file_local("review_data.parquet"))
    review_df["restaurant_id"] = review_df["restaurant_id"].astype(str)
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

    with open(_resolve_data_file_local("normalized_macro_data.json"), "r", encoding="utf-8") as handle:
        macro_raw = json.load(handle)
    macro_data, macro_default = prepare_macro_data(macro_raw)

    rows: List[Dict[str, object]] = []
    processed_restaurants = 0

    for _, rest_row in rest_df.iterrows():
        rid = str(rest_row["restaurant_id"])
        if rid not in review_cache:
            continue

        reviews = review_cache[rid]
        years_tensor = reviews["years"].long()
        year_values = sorted({int(y) for y in years_tensor.view(-1).tolist() if int(y) > 0})
        if not year_values:
            continue

        local_min = year_values[0]
        local_max = year_values[-1]
        start_year = args.min_year if args.min_year is not None else local_min
        end_year = args.max_year if args.max_year is not None else local_max
        start_year = max(start_year, local_min)
        end_year = min(end_year, local_max)
        if start_year > end_year:
            continue

        region_key = rest_row.get("region_code") if isinstance(rest_row.get("region_code"), str) else None
        name = rest_row.get("name") or rest_row.get("restaurant_name") or ""

        with torch.no_grad():
            for year in range(start_year, end_year + 1):
                batch = _build_batch_for_year(year, reviews, macro_data, macro_default, region_key)
                for key, value in batch.items():
                    batch[key] = value.to(device)
                logits = student(batch)
                if calibrator is not None:
                    logits = calibrator(logits)
                prob = torch.sigmoid(logits).detach().cpu().item()

                rows.append(
                    {
                        "restaurant_id": rid,
                        "name": name,
                        "region_code": region_key,
                        "year": year,
                        "pred_death_prob": float(prob),
                    }
                )

        processed_restaurants += 1
        if args.max_restaurants is not None and processed_restaurants >= args.max_restaurants:
            break

    if not rows:
        raise RuntimeError("No predictions produced; please check data files and year range settings.")

    base_df = pd.DataFrame(rows)

    if args.csv_time_step == "year":
        out_df = base_df
    else:
        step = 0.5 if args.csv_time_step == "half-year" else 0.25
        fine_rows: List[Dict[str, object]] = []
        for rid, sub_df in base_df.groupby("restaurant_id", sort=False):
            sub_df = sub_df.sort_values("year")
            years = sub_df["year"].astype(float).to_numpy()
            probs = sub_df["pred_death_prob"].astype(float).to_numpy()
            if len(years) == 0:
                continue
            grid = np.arange(years[0], years[-1] + 1e-8, step)
            grid_probs = np.interp(grid, years, probs)
            name = sub_df["name"].iloc[0]
            region_code = sub_df["region_code"].iloc[0]
            for t, p in zip(grid, grid_probs):
                fine_rows.append(
                    {
                        "restaurant_id": rid,
                        "name": name,
                        "region_code": region_code,
                        "year": float(t),
                        "pred_death_prob": float(p),
                    }
                )
        out_df = pd.DataFrame(fine_rows)

    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        f"Saved predictions for {out_df['restaurant_id'].nunique()} restaurants "
        f"and {len(out_df)} (restaurant, year) rows to: {args.output}"
    )

    if (args.plot_restaurant_id or args.plot_all) and plt is None:
        print("matplotlib is not installed; skip plotting. Install matplotlib to enable plotting from predict_student_all.")
        return

    base_dir = args.plot_output_dir or os.path.dirname(args.output) or "."

    if args.plot_restaurant_id:
        _plot_restaurant_curve(
            base_df,
            args.plot_restaurant_id,
            base_dir,
            plot_time_step=args.plot_time_step,
        )

    if args.plot_all:
        unique_ids = base_df["restaurant_id"].astype(str).unique().tolist()
        print(f"Plotting curves for {len(unique_ids)} restaurants to directory: {base_dir}")
        for rid in unique_ids:
            _plot_restaurant_curve(
                base_df,
                rid,
                base_dir,
                plot_time_step=args.plot_time_step,
            )


if __name__ == "__main__":
    main()
