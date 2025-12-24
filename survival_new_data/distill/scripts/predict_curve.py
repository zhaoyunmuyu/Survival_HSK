"""按离散时间步长（year/half/quarter）逐 period 推理单个餐厅的“关闭概率”序列。

说明：
- 这里的模型输出为 sigmoid(logit)，脚本以 `prob_closed` 命名；
- 这不是严格意义的 hazard（除非训练目标就是 hazard），但可用于“真实逐 period 推理”（不插值）。

运行示例：
  python -m survival_new_data.distill.scripts.predict_curve ^
      --restaurant-id 123456 ^
      --model student ^
      --checkpoint checkpoints_hsk_distill/student_best.pt ^
      --time-step quarter ^
      --min-year 2018 --max-year 2024
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

import pyarrow.dataset as ds

from survival_new_data.distill.data.io import resolve_data_path
from survival_new_data.distill.data.reviews import build_bert_vector_map, build_restaurant_review_cache, prepare_reviews_clean
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
    filt = ds.field("restaurant_id") == str(restaurant_id)
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
    filt = ds.field("review_id").isin([str(x) for x in review_ids])
    df = _read_parquet_table(path, columns=cols, filter_expr=filt)
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

    macro = torch.zeros((1, 62), dtype=torch.float32)
    batch: Dict[str, torch.Tensor] = {
        "review_text": text.unsqueeze(0),
        "review_images": images.unsqueeze(0),
        "review_features": features.unsqueeze(0),
        "review_time": review_time.unsqueeze(0),
        "last2_mask": last_mask.unsqueeze(0),
        "reference_time": torch.tensor([int(period_idx)], dtype=torch.long),
        "macro_features": macro,
        "region_encoding": torch.tensor([float(region_encoding)], dtype=torch.float32),
    }
    return batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict per-period close probability for one restaurant (no interpolation).")
    p.add_argument("--restaurant-id", type=str, required=True)
    p.add_argument("--model", type=str, choices=["student", "teacher"], default="student")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="", help="可选：survival_hsk/data 目录（默认自动定位）")
    p.add_argument("--time-step", type=str, choices=["year", "half", "quarter"], default="year")
    p.add_argument("--min-year", type=int, default=None, help="起始年份（默认按评论最早年份）")
    p.add_argument("--max-year", type=int, default=None, help="结束年份（默认按评论最晚年份）")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--text-dim", type=int, default=768)
    p.add_argument("--img-dim", type=int, default=512)
    p.add_argument("--feature-dim", type=int, default=8)
    p.add_argument("--max-offset-years", type=int, default=10)
    p.add_argument("--last-window-years", type=int, default=2)
    p.add_argument("--output", type=str, default="", help="输出 CSV 路径（默认 distill_curve_<id>.csv）")
    p.add_argument("--plot", action="store_true", help="若安装了 matplotlib，则输出折线图 PNG")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    time_step: TimeStep = args.time_step
    data_dir = Path(args.data_dir) if args.data_dir else None

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    row = _load_restaurant_row(args.restaurant_id, data_dir=data_dir)
    region_encoding = _region_encoding_from_row(row)

    # 读取该餐厅评论（parquet 过滤）
    review_df_raw = _load_reviews_for_restaurant(args.restaurant_id, data_dir=data_dir)
    review_df = prepare_reviews_clean(review_df_raw, time_step=time_step)
    review_df = review_df.sort_values("review_date", kind="mergesort")

    # 限长（与训练一致：随机采样 + 再按时间排序）
    max_reviews = 128
    if len(review_df) > max_reviews:
        review_df = review_df.sample(n=max_reviews, random_state=42).sort_values("review_date", kind="mergesort")

    review_ids = review_df["review_id"].astype(str).tolist()
    bert_df = _load_bert_vectors_for_review_ids(review_ids, data_dir=data_dir, text_dim=args.text_dim)
    bert_map = build_bert_vector_map(bert_df, text_dim=args.text_dim)

    cache = build_restaurant_review_cache(
        review_df,
        bert_map,
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

    if args.model == "student":
        model = BiLSTMStudent(
            d_model=args.hidden_dim,
            text_dim=args.text_dim,
            img_dim=args.img_dim,
            feature_dim=args.feature_dim,
            time_step=time_step,
            max_offset_years=args.max_offset_years,
        )
    else:
        model = MambaTeacher(
            d_model=args.hidden_dim,
            text_dim=args.text_dim,
            img_dim=args.img_dim,
            feature_dim=args.feature_dim,
            time_step=time_step,
            max_offset_years=args.max_offset_years,
        )

    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("state", state))
    model.to(device)
    model.eval()

    out_rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for p in _iter_periods(min_year, max_year, time_step=time_step):
            batch = _build_batch_for_period(
                period_idx=p.idx,
                reviews=reviews,
                region_encoding=region_encoding,
                time_step=time_step,
                last_window_years=args.last_window_years,
            )
            for k, v in batch.items():
                batch[k] = v.to(device)
            logits = model(batch)
            prob = float(torch.sigmoid(logits).detach().cpu().item())
            out_rows.append(
                {
                    "restaurant_id": str(args.restaurant_id),
                    "time_step": time_step,
                    "period_idx": int(p.idx),
                    "period": p.label(time_step),
                    "year": int(p.year),
                    "sub": int(p.sub),
                    "prob_closed": prob,
                    "prob_open": 1.0 - prob,
                }
            )

    out_df = pd.DataFrame(out_rows)
    print(f"restaurant_id={args.restaurant_id} model={args.model} time_step={time_step} rows={len(out_df)}")
    print(out_df[["period", "prob_closed"]].head(12).to_string(index=False))

    out_path = args.output or f"distill_curve_{args.restaurant_id}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved:", out_path)

    if args.plot and plt is not None:
        fig_path = Path(out_path).with_suffix(".png")
        xs = np.arange(len(out_df), dtype=float)
        ys = out_df["prob_closed"].astype(float).to_numpy()
        plt.figure(figsize=(8, 3))
        plt.plot(xs, ys)
        plt.xticks(xs[:: max(1, len(xs) // 10)], out_df["period"].iloc[:: max(1, len(xs) // 10)].tolist(), rotation=45, ha="right")
        plt.ylabel("prob_closed")
        plt.title(f"{args.model} prob_closed - {args.restaurant_id} ({time_step})")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        print("saved plot:", str(fig_path))
    elif args.plot:
        print("matplotlib not installed; skip plotting")


if __name__ == "__main__":
    main()
