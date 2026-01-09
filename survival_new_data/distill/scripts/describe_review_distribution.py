"""查看单个餐厅的评论时间分布（year/half/quarter）。

示例：
  python -m survival_new_data.distill.scripts.describe_review_distribution ^
    --restaurant-id 90 ^
    --data-dir /nvme01/gvm0/hsk/data ^
    --show-sampled
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from survival_new_data.distill.data.io import resolve_data_path
from survival_new_data.distill.data.reviews import prepare_reviews_clean


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


def _field_str(name: str):
    return ds.field(name).cast(pa.string())


def _read_parquet_table(path: Path, *, columns: Optional[List[str]] = None, filter_expr=None) -> pd.DataFrame:
    dataset = ds.dataset(str(path), format="parquet")
    table = dataset.to_table(columns=columns, filter=filter_expr)
    return table.to_pandas()


def _periods_per_year(time_step: TimeStep) -> int:
    return {"year": 1, "half": 2, "quarter": 4}[time_step]


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


def _add_period_label(df: pd.DataFrame, *, time_step: TimeStep) -> pd.DataFrame:
    if "review_time_idx" not in df.columns:
        raise KeyError("Missing column: review_time_idx (did you call prepare_reviews_clean?)")
    labels = [_period_from_index(int(x), time_step=time_step).label(time_step) if int(x) > 0 else "UNKNOWN" for x in df["review_time_idx"]]
    out = df.copy()
    out["period"] = labels
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Describe one restaurant review date distribution by year/half/quarter.")
    p.add_argument("--restaurant-id", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="", help="可选：survival_hsk/data 目录（默认自动定位）")
    p.add_argument("--time-step", type=str, choices=["year", "half", "quarter"], default="year")
    p.add_argument("--max-reviews", type=int, default=128, help="对齐模型的 max_reviews（用于 sampled 统计）")
    p.add_argument("--show-sampled", action="store_true", help="同时输出“按模型采样后”的分布（随机种子=42）")
    p.add_argument("--output", type=str, default="", help="可选：输出分布 CSV（包含 full 与 sampled）")
    return p.parse_args()


def _summarize(df: pd.DataFrame, *, time_step: TimeStep) -> pd.DataFrame:
    df2 = _add_period_label(df, time_step=time_step)
    counts = df2.groupby("period", sort=False).size().rename("count").reset_index()
    # 让 UNKNOWN 放最后，其他按 period_idx 的自然顺序
    if "UNKNOWN" in set(counts["period"]):
        known = counts[counts["period"] != "UNKNOWN"].copy()
        unk = counts[counts["period"] == "UNKNOWN"].copy()
        counts = pd.concat([known, unk], ignore_index=True)
    return counts


def main() -> None:
    args = parse_args()
    time_step: TimeStep = args.time_step
    data_dir = Path(args.data_dir) if args.data_dir else None

    reviews_path = resolve_data_path("reviews_clean.parquet", data_dir=data_dir)
    if not reviews_path.exists():
        raise FileNotFoundError(f"reviews_clean.parquet not found at: {reviews_path}")

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
    filt = _field_str("restaurant_id") == str(args.restaurant_id)
    df = _read_parquet_table(reviews_path, columns=cols, filter_expr=filt)
    if df.empty:
        raise RuntimeError(f"No reviews found for restaurant_id={args.restaurant_id!r} in reviews_clean.parquet")

    df["restaurant_id"] = df["restaurant_id"].astype(str)
    df["review_id"] = df["review_id"].astype(str)
    df = prepare_reviews_clean(df, time_step=time_step)  # type: ignore[arg-type]
    df = df.sort_values("review_date", kind="mergesort")

    valid_dates = df["review_date"].dropna()
    print(f"restaurant_id={args.restaurant_id}")
    print(f"reviews_total={len(df)}")
    if not valid_dates.empty:
        print(f"date_range={valid_dates.min().date()}..{valid_dates.max().date()}")
    else:
        print("date_range=UNKNOWN")

    full_counts = _summarize(df, time_step=time_step)
    print("\n[full] counts by period:")
    print(full_counts.to_string(index=False))

    sampled_counts = None
    max_reviews = int(args.max_reviews)
    if args.show_sampled:
        if len(df) > max_reviews:
            df_sampled = df.sample(n=max_reviews, random_state=42).sort_values("review_date", kind="mergesort")
        else:
            df_sampled = df
        sampled_counts = _summarize(df_sampled, time_step=time_step)
        print(f"\n[sampled] counts by period (max_reviews={max_reviews}, seed=42):")
        print(sampled_counts.to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full_out = full_counts.copy()
        full_out["subset"] = "full"
        if sampled_counts is not None:
            samp_out = sampled_counts.copy()
            samp_out["subset"] = "sampled"
            out_df = pd.concat([full_out, samp_out], ignore_index=True)
        else:
            out_df = full_out
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

