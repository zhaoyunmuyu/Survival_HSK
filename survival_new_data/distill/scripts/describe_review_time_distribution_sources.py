"""查看 OpenRice CSV 与 review_data.parquet 中，单个餐厅的评论时间分布。

示例：
  python -m survival_new_data.distill.scripts.describe_review_time_distribution_sources ^
    --restaurant-id 13673 ^
    --freq month ^
    --output-dir artifacts
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


@dataclass(frozen=True)
class SourceSummary:
    name: str
    n_reviews: int
    date_min: str
    date_max: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Describe review_date distribution for one restaurant across sources.")
    p.add_argument("--restaurant-id", type=str, required=True)
    p.add_argument("--openrice-csv", type=str, default=str(Path("openrice") / "整合餐厅评论信息.csv"))
    p.add_argument("--parquet", type=str, default="review_data.parquet")
    p.add_argument("--freq", type=str, choices=["year", "quarter", "month", "week"], default="month")
    p.add_argument("--chunksize", type=int, default=200_000, help="Chunk size when reading OpenRice CSV.")
    p.add_argument("--output-dir", type=str, default="", help="Optional: directory to save distribution CSVs.")
    return p.parse_args()


def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _bucket_label(dt: pd.Series, *, freq: str) -> pd.Series:
    if freq == "year":
        return dt.dt.year.astype("Int64").astype(str)
    if freq == "quarter":
        return dt.dt.to_period("Q").astype(str)
    if freq == "month":
        return dt.dt.to_period("M").astype(str)
    if freq == "week":
        # ISO 周（年-周）
        iso = dt.dt.isocalendar()
        return (iso["year"].astype("Int64").astype(str) + "-W" + iso["week"].astype("Int64").astype(str).str.zfill(2)).astype(str)
    raise ValueError(f"Unsupported freq: {freq}")


def _summarize_dates(review_date: pd.Series, *, name: str, freq: str) -> tuple[SourceSummary, pd.Series]:
    dt = _to_datetime(review_date).dropna()
    if dt.empty:
        summary = SourceSummary(name=name, n_reviews=0, date_min="NA", date_max="NA")
        return summary, pd.Series(dtype="int64")

    labels = _bucket_label(dt, freq=freq)
    counts = labels.value_counts().sort_index()
    summary = SourceSummary(
        name=name,
        n_reviews=int(len(dt)),
        date_min=dt.min().date().isoformat(),
        date_max=dt.max().date().isoformat(),
    )
    return summary, counts


def _read_openrice_dates(csv_path: Path, *, restaurant_id: str, chunksize: int) -> pd.Series:
    usecols = ["restaurant_id", "review_date"]
    hits: list[pd.DataFrame] = []

    # 有些环境下 utf-8-sig 能更稳（BOM），失败则回退为 utf-8。
    encodings = ["utf-8-sig", "utf-8"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            for chunk in pd.read_csv(
                csv_path,
                usecols=usecols,
                dtype={"restaurant_id": "string", "review_date": "string"},
                encoding=enc,
                chunksize=int(chunksize),
            ):
                mask = chunk["restaurant_id"].astype(str) == str(restaurant_id)
                if mask.any():
                    hits.append(chunk.loc[mask, ["review_date"]])
            last_err = None
            break
        except Exception as e:  # pragma: no cover
            last_err = e
            hits = []
            continue

    if last_err is not None:  # pragma: no cover
        raise last_err

    if not hits:
        return pd.Series([], dtype="string", name="review_date")
    df = pd.concat(hits, ignore_index=True)
    return df["review_date"]


def _read_parquet_dates(parquet_path: Path, *, restaurant_id: str) -> pd.Series:
    dataset = ds.dataset(str(parquet_path), format="parquet")
    filt = ds.field("restaurant_id").cast(pa.string()) == str(restaurant_id)
    table = dataset.to_table(columns=["review_date"], filter=filt)
    df = table.to_pandas()
    if "review_date" not in df.columns:
        return pd.Series([], dtype="string", name="review_date")
    return df["review_date"]


def _print_counts(counts: pd.Series, *, max_rows: int = 24) -> None:
    if counts.empty:
        print("<empty>")
        return
    if len(counts) <= max_rows:
        print(counts.to_string())
        return
    head_n = max_rows // 2
    tail_n = max_rows - head_n
    show = pd.concat([counts.head(head_n), counts.tail(tail_n)])
    print(show.to_string())
    print(f"... total_buckets={len(counts)}")


def _maybe_save(output_dir: str, *, restaurant_id: str, source_key: str, freq: str, counts: pd.Series) -> None:
    if not output_dir:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rest_{restaurant_id}_{source_key}_{freq}_counts.csv"
    pd.DataFrame({"bucket": counts.index, "count": counts.values}).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    rid = str(args.restaurant_id)
    freq = str(args.freq)

    openrice_csv = Path(args.openrice_csv)
    parquet_path = Path(args.parquet)

    openrice_dates = _read_openrice_dates(openrice_csv, restaurant_id=rid, chunksize=int(args.chunksize))
    openrice_summary, openrice_counts = _summarize_dates(openrice_dates, name=str(openrice_csv), freq=freq)

    parquet_dates = _read_parquet_dates(parquet_path, restaurant_id=rid)
    parquet_summary, parquet_counts = _summarize_dates(parquet_dates, name=str(parquet_path), freq=freq)

    for key, summary, counts in [
        ("openrice", openrice_summary, openrice_counts),
        ("parquet", parquet_summary, parquet_counts),
    ]:
        print(f"\n== {summary.name} ==")
        print(f"restaurant_id={rid}")
        print(f"n_reviews={summary.n_reviews}")
        print(f"date_range={summary.date_min}..{summary.date_max}")
        print(f"\n[counts by {freq}]")
        _print_counts(counts)
        _maybe_save(args.output_dir, restaurant_id=rid, source_key=key, freq=freq, counts=counts)


if __name__ == "__main__":
    main()

