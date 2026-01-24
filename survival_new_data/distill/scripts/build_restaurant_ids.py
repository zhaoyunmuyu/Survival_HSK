"""生成店铺 id 列表（可按“评论年份覆盖”过滤）。

典型用途：为 `predict_survival_curve_all.py` 准备 `all_ids.txt`，只预测满足条件的店铺。

示例：
  python -m survival_new_data.distill.scripts.build_restaurant_ids ^
    --data-dir /nvme02/gvm03-data/hsk/data ^
    --out all_ids_3y.txt ^
    --min-review-years 3 ^
    --year-mode distinct
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import pandas as pd
import pyarrow.dataset as ds

from survival_new_data.distill.data.io import resolve_data_path


YearMode = Literal["distinct", "span"]


@dataclass
class _Agg:
    min_year: int
    max_year: int
    year_mask: int  # only used for distinct-mode


def _iter_batches(path: Path, *, batch_size: int) -> Iterable[pd.DataFrame]:
    dataset = ds.dataset(str(path), format="parquet")
    scanner = dataset.scanner(columns=["restaurant_id", "review_date"], batch_size=int(batch_size))
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        yield df


def _year_to_bit(year: int, *, offset: int) -> int:
    shift = int(year) - int(offset)
    if shift < 0:
        return 0
    return 1 << shift


def _popcount(x: int) -> int:
    # python3.8+ has int.bit_count()
    return int(x).bit_count()


def _try_sort_ids(ids: List[str]) -> List[str]:
    if all(str(x).isdigit() for x in ids):
        return [str(x) for x in sorted((int(x) for x in ids))]
    return sorted((str(x) for x in ids))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build restaurant id list with optional review-year coverage filtering.")
    p.add_argument("--data-dir", type=str, default="", help="数据目录（默认自动定位）")
    p.add_argument("--reviews", type=str, default="reviews_clean.parquet", help="默认：reviews_clean.parquet")
    p.add_argument("--out", type=str, default="all_ids.txt")
    p.add_argument("--min-review-years", type=int, default=3, help="至少需要覆盖多少个年份（默认3）")
    p.add_argument(
        "--year-mode",
        type=str,
        choices=["distinct", "span"],
        default="distinct",
        help="distinct=不同年份数量；span=跨度(max_year-min_year+1)",
    )
    p.add_argument("--batch-size", type=int, default=200_000)
    p.add_argument("--year-bit-offset", type=int, default=1980, help="distinct 模式用：年份 bitmask 的 offset（默认1980）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    year_mode: YearMode = args.year_mode
    min_years = int(args.min_review_years)
    if min_years <= 0:
        raise ValueError("--min-review-years must be > 0")

    data_dir = Path(args.data_dir) if args.data_dir else None
    reviews_path = resolve_data_path(str(args.reviews), data_dir=data_dir)
    if not reviews_path.exists():
        raise FileNotFoundError(f"reviews parquet not found: {reviews_path}")

    out_path = Path(args.out)
    agg: Dict[str, _Agg] = {}

    total_rows = 0
    bad_date_rows = 0

    for df in _iter_batches(reviews_path, batch_size=int(args.batch_size)):
        if df.empty:
            continue
        total_rows += int(len(df))

        # restaurant_id
        df["restaurant_id"] = df["restaurant_id"].astype(str)

        # robust year extraction
        dt = pd.to_datetime(df["review_date"], errors="coerce")
        years = dt.dt.year
        ok = years.notna()
        if not bool(ok.all()):
            bad_date_rows += int((~ok).sum())
        df = df.loc[ok, ["restaurant_id"]].copy()
        df["year"] = years.loc[ok].astype(int).to_numpy(copy=False)
        if df.empty:
            continue

        grouped = df.groupby("restaurant_id")["year"]
        for rid, y in grouped:
            y_min = int(y.min())
            y_max = int(y.max())
            if year_mode == "distinct":
                uniq_years = y.unique().tolist()
                mask = 0
                for yy in uniq_years:
                    mask |= _year_to_bit(int(yy), offset=int(args.year_bit_offset))
            else:
                mask = 0

            prev = agg.get(rid)
            if prev is None:
                agg[rid] = _Agg(min_year=y_min, max_year=y_max, year_mask=mask)
            else:
                prev.min_year = min(int(prev.min_year), y_min)
                prev.max_year = max(int(prev.max_year), y_max)
                if year_mode == "distinct":
                    prev.year_mask |= mask

    kept: List[str] = []
    for rid, a in agg.items():
        if year_mode == "distinct":
            years_cnt = _popcount(int(a.year_mask))
        else:
            years_cnt = int(a.max_year) - int(a.min_year) + 1
        if years_cnt >= min_years:
            kept.append(str(rid))

    kept = _try_sort_ids(kept)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    print(
        f"[HSK][BuildIDs] reviews={reviews_path} rows={total_rows} restaurants={len(agg)} "
        f"kept={len(kept)} min_years={min_years} mode={year_mode} bad_date_rows={bad_date_rows} out={out_path}"
    )


if __name__ == "__main__":
    main()

