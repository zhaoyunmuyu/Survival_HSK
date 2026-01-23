"""按离散时间步（year/half/quarter）对单个餐厅做“生存/关店概率”折线图推理（不插值）。

输出：
- CSV：每个 period 一行，包含 prob_closed 与 prob_survival(=1-prob_closed)
- （可选）PNG 折线图：默认画 prob_survival

示例：
  python -m survival_new_data.distill.scripts.predict_survival_curve ^
    --restaurant-id 123456 ^
    --model student ^
    --checkpoint checkpoints_hsk_distill/student_best.pt ^
    --time-step quarter --min-year 2018 --max-year 2024 ^
    --plot
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

import pyarrow as pa
import pyarrow.dataset as ds

from survival_new_data.distill.data.io import resolve_data_path
from survival_new_data.distill.data.reviews import (
    build_bert_vector_map,
    build_img_vector_map,
    build_restaurant_review_cache,
    prepare_reviews_clean,
)
from survival_new_data.distill.models import BiLSTMStudent, MambaTeacher


TimeStep = Literal["year", "half", "quarter"]


@dataclass(frozen=True)
class Period:
    idx: int
    year: int
    sub: int  # quarter=1..4, half=1..2, year=1

    def label(self, time_step: TimeStep) -> str:
        if time_step == "year":
            return f"{self.year}"
        if time_step == "half":
            return f"{self.year}-H{self.sub}"
        return f"{self.year}Q{self.sub}"


def _periods_per_year(time_step: TimeStep) -> int:
    return {"year": 1, "half": 2, "quarter": 4}[time_step]


def _period_index_from_year_sub(year: int, sub: int, *, time_step: TimeStep) -> int:
    if time_step == "year":
        return int(year)
    if time_step == "half":
        return int(year) * 2 + (int(sub) - 1)
    return int(year) * 4 + (int(sub) - 1)


def _period_from_index(idx: int, *, time_step: TimeStep) -> Period:
    if time_step == "year":
        return Period(idx=int(idx), year=int(idx), sub=1)
    if time_step == "half":
        year = int(idx) // 2
        half = int(idx) % 2 + 1
        return Period(idx=int(idx), year=year, sub=half)
    year = int(idx) // 4
    quarter = int(idx) % 4 + 1
    return Period(idx=int(idx), year=year, sub=quarter)


def _iter_periods(min_year: int, max_year: int, *, time_step: TimeStep) -> Iterable[Period]:
    ppy = _periods_per_year(time_step)
    for y in range(int(min_year), int(max_year) + 1):
        for sub in range(1, ppy + 1):
            idx = _period_index_from_year_sub(y, sub, time_step=time_step)
            yield _period_from_index(idx, time_step=time_step)


def _read_parquet_table(path: Path, *, columns: Optional[List[str]] = None, filter_expr=None) -> pd.DataFrame:
    dataset = ds.dataset(str(path), format="parquet")
    table = dataset.to_table(columns=columns, filter=filter_expr)
    return table.to_pandas()


def _field_str(name: str):
    """pyarrow.dataset 的过滤表达式要求左右类型一致；统一把字段 cast 成 string 便于与命令行参数对齐。"""
    return ds.field(name).cast(pa.string())


def _load_restaurant_row(restaurant_id: str, *, data_dir: Optional[Path]) -> Dict[str, object]:
    path = resolve_data_path("restaurant_base.parquet", data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"restaurant_base.parquet not found at: {path}")
    df = pd.read_parquet(path)
    df["restaurant_id"] = df["restaurant_id"].astype(str)
    row_df = df[df["restaurant_id"] == restaurant_id]
    if row_df.empty:
        raise KeyError(f"restaurant_id {restaurant_id!r} not found in restaurant_base.parquet")
    return row_df.iloc[0].to_dict()


def _load_reviews_for_restaurant(restaurant_id: str, *, data_dir: Optional[Path]) -> pd.DataFrame:
    path = resolve_data_path("reviews_clean.parquet", data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"reviews_clean.parquet not found at: {path}")
    cols = [
        "restaurant_id",
        "review_id",
        "review_date",
        "user_location",
        "review_helpful_vote",
        "review_taste",
        "review_environment",
        "review_service",
        "review_hygiene",
        "review_dishi",
    ]
    filt = _field_str("restaurant_id") == str(restaurant_id)
    df = _read_parquet_table(path, columns=cols, filter_expr=filt)
    if df.empty:
        raise RuntimeError(f"No reviews found for restaurant_id={restaurant_id!r} in reviews_clean.parquet")
    df["restaurant_id"] = df["restaurant_id"].astype(str)
    df["review_id"] = df["review_id"].astype(str)
    return df


def _load_bert_vectors_for_review_ids(review_ids: List[str], *, data_dir: Optional[Path], text_dim: int) -> pd.DataFrame:
    path = resolve_data_path("review_bert_emb.parquet", data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"review_bert_emb.parquet not found at: {path}")

    cols = ["review_id"] + [f"bert_emb_{i}" for i in range(int(text_dim))]
    filt = _field_str("review_id").isin([str(x) for x in review_ids])
    df = _read_parquet_table(path, columns=cols, filter_expr=filt)
    df["review_id"] = df["review_id"].astype(str)
    return df


def _load_img_vectors_for_review_ids(review_ids: List[str], *, data_dir: Optional[Path], img_dim: int) -> Optional[pd.DataFrame]:
    path = resolve_data_path("review_img_emb.parquet", data_dir=data_dir)
    if not path.exists():
        parts_dir = Path(str(path.with_suffix("")) + ".parts")
        if parts_dir.exists() and parts_dir.is_dir():
            path = parts_dir
        else:
            return None

    cols = ["review_id"] + [f"img_emb_{i}" for i in range(int(img_dim))]
    filt = _field_str("review_id").isin([str(x) for x in review_ids])
    df = _read_parquet_table(path, columns=cols, filter_expr=filt)
    if df.empty:
        return df
    df["review_id"] = df["review_id"].astype(str)
    return df


def _region_encoding_from_row(row: Dict[str, object]) -> float:
    v = row.get("restaurant_region_num")
    try:
        num = float(v)  # type: ignore[arg-type]
    except Exception:
        return 0.0
    if num <= 0:
        return 0.0
    return float(min(num, 18.0))


def _build_batch_for_period(
    *,
    period_idx: int,
    reviews: Dict[str, torch.Tensor],
    region_encoding: float,
    time_step: TimeStep,
    last_window_years: int,
) -> Dict[str, torch.Tensor]:
    text = reviews["text"]  # [L, d]
    images = reviews["images"]  # [L, img]
    features = reviews["features"]  # [L, f]
    review_time = reviews["years"].long()  # [L]

    periods = _periods_per_year(time_step)
    last_window_periods = int(last_window_years) * periods
    keep_from = int(period_idx) - (last_window_periods - 1)
    last_mask = (review_time >= keep_from) & (review_time > 0)

    return {
        "review_text": text.unsqueeze(0),
        "review_images": images.unsqueeze(0),
        "review_features": features.unsqueeze(0),
        "review_time": review_time.unsqueeze(0),
        "last2_mask": last_mask.unsqueeze(0),
        "reference_time": torch.tensor([int(period_idx)], dtype=torch.long),
        "region_encoding": torch.tensor([float(region_encoding)], dtype=torch.float32),
    }


def _load_model(
    *,
    model_name: Literal["student", "teacher"],
    checkpoint: str,
    hidden_dim: int,
    text_dim: int,
    img_dim: int,
    feature_dim: int,
    time_step: TimeStep,
    max_offset_years: int,
    device: torch.device,
) -> torch.nn.Module:
    if model_name == "student":
        model = BiLSTMStudent(
            d_model=hidden_dim,
            text_dim=text_dim,
            img_dim=img_dim,
            feature_dim=feature_dim,
            time_step=time_step,
            max_offset_years=max_offset_years,
        )
    else:
        model = MambaTeacher(
            d_model=hidden_dim,
            text_dim=text_dim,
            img_dim=img_dim,
            feature_dim=feature_dim,
            time_step=time_step,
            max_offset_years=max_offset_years,
        )

    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("state", state))
    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict per-period survival/close probabilities for one restaurant (no interpolation).")
    p.add_argument("--restaurant-id", type=str, required=True)
    p.add_argument("--model", type=str, choices=["student", "teacher"], default="student")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="", help="可选：survival_hsk/data 目录（默认自动定位）")
    p.add_argument("--time-step", type=str, choices=["year", "half", "quarter"], default="year")
    p.add_argument("--min-year", type=int, default=None)
    p.add_argument("--max-year", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--text-dim", type=int, default=768)
    p.add_argument("--img-dim", type=int, default=512)
    p.add_argument("--feature-dim", type=int, default=8)
    p.add_argument("--max-offset-years", type=int, default=10)
    p.add_argument("--last-window-years", type=int, default=2)
    p.add_argument("--output", type=str, default="", help="输出 CSV 路径（默认 survival_curve_<id>.csv）")
    p.add_argument("--plot", action="store_true", help="若安装 matplotlib，则输出折线图 PNG（画 prob_survival）")
    p.add_argument(
        "--plot-y",
        type=str,
        choices=["survival", "closed"],
        default="survival",
        help="画哪条曲线：survival=1-prob_closed，closed=prob_closed（默认 survival）",
    )
    p.add_argument(
        "--max-xticks",
        type=int,
        default=24,
        help="折线图最多显示多少个 x 轴刻度（用于避免 quarter 太密），默认 24",
    )
    p.add_argument(
        "--xtick-year-only",
        action="store_true",
        help="只在整年边界显示 x 轴刻度（quarter 显示 Q1，half 显示 H1），并用年份作为标签",
    )
    p.add_argument("--line-width", type=float, default=1.5, help="折线宽度（默认 1.5）")
    p.add_argument("--marker-size", type=float, default=2.0, help="点的大小（markersize，默认 2.0）")
    p.add_argument("--marker-every", type=int, default=1, help="每隔多少个点画一个 marker（默认每个点都画）")
    p.add_argument(
        "--plot-smooth-window",
        type=int,
        default=1,
        help="仅对绘图曲线做滑动均值平滑（不影响 CSV），窗口大小（默认 1=不平滑）",
    )
    p.add_argument("--fig-width", type=float, default=0.0, help="图宽（英寸），0=自动")
    p.add_argument("--fig-height", type=float, default=4.0, help="图高（英寸），默认 4.0")
    p.add_argument("--fig-width-per-point", type=float, default=0.22, help="自动图宽时每个点贡献的宽度（越小越紧凑）")
    p.add_argument("--fig-min-width", type=float, default=8.0, help="自动图宽最小值")
    p.add_argument("--fig-max-width", type=float, default=14.0, help="自动图宽最大值")
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    time_step: TimeStep = args.time_step
    data_dir = Path(args.data_dir) if args.data_dir else None

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    row = _load_restaurant_row(args.restaurant_id, data_dir=data_dir)
    region_encoding = _region_encoding_from_row(row)

    review_df_raw = _load_reviews_for_restaurant(args.restaurant_id, data_dir=data_dir)
    review_df = prepare_reviews_clean(review_df_raw, time_step=time_step)
    review_df = review_df.sort_values("review_date", kind="mergesort")

    max_reviews = 128
    if len(review_df) > max_reviews:
        review_df = review_df.sample(n=max_reviews, random_state=42).sort_values("review_date", kind="mergesort")

    review_ids = review_df["review_id"].astype(str).tolist()
    bert_df = _load_bert_vectors_for_review_ids(review_ids, data_dir=data_dir, text_dim=args.text_dim)
    bert_map = build_bert_vector_map(bert_df, text_dim=args.text_dim)

    img_map = None
    img_df = _load_img_vectors_for_review_ids(review_ids, data_dir=data_dir, img_dim=args.img_dim)
    if img_df is not None and not img_df.empty:
        img_map = build_img_vector_map(img_df, img_dim=args.img_dim)

    cache = build_restaurant_review_cache(
        review_df,
        bert_map,
        img_map,
        max_reviews=max_reviews,
        text_dim=args.text_dim,
        img_dim=args.img_dim,
        feature_dim=args.feature_dim,
    )
    reviews = cache.get(str(args.restaurant_id))
    if reviews is None:
        raise RuntimeError("Failed to build review cache for restaurant")

    time_values = reviews["years"].long().view(-1)
    valid = time_values[time_values > 0]
    if valid.numel() == 0:
        raise RuntimeError("No valid review_time_idx found for this restaurant")

    min_period = int(valid.min().item())
    max_period = int(valid.max().item())
    min_year_auto = _period_from_index(min_period, time_step=time_step).year
    max_year_auto = _period_from_index(max_period, time_step=time_step).year
    min_year = int(args.min_year) if args.min_year is not None else int(min_year_auto)
    max_year = int(args.max_year) if args.max_year is not None else int(max_year_auto)
    if min_year > max_year:
        raise ValueError(f"min-year ({min_year}) > max-year ({max_year})")

    model = _load_model(
        model_name=args.model,
        checkpoint=args.checkpoint,
        hidden_dim=int(args.hidden_dim),
        text_dim=int(args.text_dim),
        img_dim=int(args.img_dim),
        feature_dim=int(args.feature_dim),
        time_step=time_step,
        max_offset_years=int(args.max_offset_years),
        device=device,
    )

    rows_out: List[Dict[str, object]] = []
    with torch.no_grad():
        for period in _iter_periods(min_year, max_year, time_step=time_step):
            batch = _build_batch_for_period(
                period_idx=period.idx,
                reviews=reviews,
                region_encoding=region_encoding,
                time_step=time_step,
                last_window_years=int(args.last_window_years),
            )
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            logits = model(batch)
            prob_closed = float(torch.sigmoid(logits).detach().cpu().view(-1).item())
            rows_out.append(
                {
                    "period_idx": int(period.idx),
                    "period": period.label(time_step),
                    "prob_closed": prob_closed,
                    "prob_survival": 1.0 - prob_closed,
                }
            )

    df_out = pd.DataFrame(rows_out)
    output = args.output or f"survival_curve_{args.restaurant_id}.csv"
    df_out.to_csv(output, index=False)
    print(f"Saved: {output}")

    if args.plot:
        if plt is None:
            raise RuntimeError("matplotlib is not available, cannot plot")
        y_key = "prob_survival" if args.plot_y == "survival" else "prob_closed"
        y_label = y_key

        # x 轴太密（尤其是 quarter）时，改用索引轴，并稀疏显示刻度标签
        n = len(df_out)
        x = np.arange(n)
        max_xticks = max(2, int(args.max_xticks))
        if args.xtick_year_only:
            periods = df_out["period"].astype(str).tolist()
            year_pos: List[int] = []
            year_labels: List[str] = []
            for i, label in enumerate(periods):
                if time_step == "year":
                    ok = True
                elif time_step == "half":
                    ok = label.endswith("-H1")
                else:  # quarter
                    ok = label.endswith("Q1")
                if ok:
                    year_pos.append(i)
                    year_labels.append(label[:4])

            if not year_pos:
                step = max(1, int(np.ceil(n / max_xticks)))
                tick_pos = np.arange(0, n, step, dtype=int)
                tick_labels = df_out["period"].iloc[tick_pos].tolist()
            else:
                step = max(1, int(np.ceil(len(year_pos) / max_xticks)))
                tick_pos = np.asarray(year_pos[::step], dtype=int)
                tick_labels = [year_labels[i] for i in range(0, len(year_labels), step)]
        else:
            step = max(1, int(np.ceil(n / max_xticks)))
            tick_pos = np.arange(0, n, step, dtype=int)
            tick_labels = df_out["period"].iloc[tick_pos].tolist()

        if float(args.fig_width) > 0:
            fig_w = float(args.fig_width)
        else:
            fig_w = float(np.clip(n * float(args.fig_width_per_point), float(args.fig_min_width), float(args.fig_max_width)))
        fig = plt.figure(figsize=(fig_w, float(args.fig_height)))
        ax = fig.add_subplot(1, 1, 1)
        y = df_out[y_key].to_numpy(dtype=float, copy=False)
        smooth_w = max(1, int(args.plot_smooth_window))
        if smooth_w > 1:
            y = pd.Series(y).rolling(window=smooth_w, center=True, min_periods=1).mean().to_numpy()

        marker_every = max(1, int(args.marker_every))
        ax.plot(
            x,
            y,
            marker="o",
            markersize=float(args.marker_size),
            markevery=marker_every,
            linewidth=float(args.line_width),
        )
        ax.set_title(f"Survival probability curve ({args.model}) - restaurant_id={args.restaurant_id}")
        ax.set_xlabel("period")
        ax.set_ylabel(y_label)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        fig.tight_layout()
        png_path = Path(output).with_suffix(".png")
        fig.savefig(png_path, dpi=150)
        print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
