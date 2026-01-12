"""批量为多个餐厅生成生存/关店概率折线图（不插值）。

默认读取：
- logs_ids/split_train_all.txt
- logs_ids/split_test_all.txt

并输出到：
- plots_survival_curves/
  - <split>/survival_curve_<id>.csv
  - <split>/survival_curve_<id>.png  （需要 --plot）

示例：
  python -m survival_new_data.distill.scripts.predict_survival_curve_all ^
    --model student ^
    --checkpoint checkpoints_hsk_distill/student_best.pt ^
    --data-dir /nvme01/gvm0/hsk/data ^
    --time-step quarter ^
    --plot --xtick-year-only --max-xticks 10 ^
    --marker-size 1.0 --marker-every 4 --plot-smooth-window 5 ^
    --fig-width 7 --fig-height 3.5
"""
python -m survival_new_data.distill.scripts.predict_survival_curve_all --model student --checkpoint checkpoints_hsk_distill/student_best.pt --data-dir /nvme01/gvm0/hsk/data --time-step quarter --out-dir plots_survival_curves --plot --xtick-year-only --max-xticks 10 --marker-size 1.0 --marker-every 4 --plot-smooth-window 5 --fig-width 7 --fig-height 3.5
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


def _field_str(name: str):
    return ds.field(name).cast(pa.string())


def _read_parquet_table(path: Path, *, columns: Optional[List[str]] = None, filter_expr=None) -> pd.DataFrame:
    dataset = ds.dataset(str(path), format="parquet")
    table = dataset.to_table(columns=columns, filter=filter_expr)
    return table.to_pandas()


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


def _load_restaurant_region(restaurant_id: str, *, data_dir: Optional[Path]) -> float:
    path = resolve_data_path("restaurant_base.parquet", data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(f"restaurant_base.parquet not found at: {path}")
    cols = ["restaurant_id", "restaurant_region_num"]
    filt = _field_str("restaurant_id") == str(restaurant_id)
    df = _read_parquet_table(path, columns=cols, filter_expr=filt)
    if df.empty:
        raise KeyError(f"restaurant_id {restaurant_id!r} not found in restaurant_base.parquet")
    v = df.iloc[0].get("restaurant_region_num")
    try:
        num = float(v)
    except Exception:
        return 0.0
    if num <= 0:
        return 0.0
    return float(min(num, 18.0))


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
        return None

    cols = ["review_id"] + [f"img_emb_{i}" for i in range(int(img_dim))]
    filt = _field_str("review_id").isin([str(x) for x in review_ids])
    df = _read_parquet_table(path, columns=cols, filter_expr=filt)
    if df.empty:
        return df
    df["review_id"] = df["review_id"].astype(str)
    return df


def _build_batch_for_period(
    *,
    period_idx: int,
    reviews: Dict[str, torch.Tensor],
    region_encoding: float,
    time_step: TimeStep,
    last_window_years: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    text = reviews["text"]
    images = reviews["images"]
    features = reviews["features"]
    review_time = reviews["years"].long()

    periods = _periods_per_year(time_step)
    last_window_periods = int(last_window_years) * periods
    keep_from = int(period_idx) - (last_window_periods - 1)
    last_mask = (review_time >= keep_from) & (review_time > 0)

    batch: Dict[str, torch.Tensor] = {
        "review_text": text.unsqueeze(0).to(device),
        "review_images": images.unsqueeze(0).to(device),
        "review_features": features.unsqueeze(0).to(device),
        "review_time": review_time.unsqueeze(0).to(device),
        "last2_mask": last_mask.unsqueeze(0).to(device),
        "reference_time": torch.tensor([int(period_idx)], dtype=torch.long, device=device),
        "region_encoding": torch.tensor([float(region_encoding)], dtype=torch.float32, device=device),
    }
    return batch


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


def _read_ids_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"ids file not found: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # 兼容 logs_ids 文件：第一行可能是示例命令
    if lines and (lines[0].startswith("python ") or lines[0].startswith("python\t") or " -m " in lines[0]):
        lines = lines[1:]
    return lines


def _plot_curve(
    df_out: pd.DataFrame,
    *,
    out_png: Path,
    title: str,
    time_step: TimeStep,
    y_key: str,
    max_xticks: int,
    xtick_year_only: bool,
    plot_x: str,
    year_reducer: str,
    line_width: float,
    marker_size: float,
    marker_every: int,
    smooth_window: int,
    fig_width: float,
    fig_height: float,
    fig_width_per_point: float,
    fig_min_width: float,
    fig_max_width: float,
) -> None:
    if plt is None:
        return

    plot_df = df_out.copy()
    if plot_x == "year":
        plot_df["year"] = plot_df["period"].astype(str).str.slice(0, 4)
        if year_reducer == "first":
            plot_df = plot_df.groupby("year", sort=True, as_index=False).first(numeric_only=False)
        elif year_reducer == "last":
            plot_df = plot_df.groupby("year", sort=True, as_index=False).last(numeric_only=False)
        else:
            plot_df = plot_df.groupby("year", sort=True, as_index=False).mean(numeric_only=True)
            plot_df["period"] = plot_df["year"]

    n = len(plot_df)
    x = np.arange(n)
    max_xticks = max(2, int(max_xticks))

    if plot_x == "year":
        tick_pos = np.arange(n, dtype=int)
        tick_labels = plot_df["period"].astype(str).tolist()
    elif xtick_year_only:
        periods = plot_df["period"].astype(str).tolist()
        year_pos: List[int] = []
        year_labels: List[str] = []
        for i, label in enumerate(periods):
            if time_step == "year":
                ok = True
            elif time_step == "half":
                ok = label.endswith("-H1")
            else:
                ok = label.endswith("Q1")
            if ok:
                year_pos.append(i)
                year_labels.append(label[:4])
        if not year_pos:
            step = max(1, int(np.ceil(n / max_xticks)))
            tick_pos = np.arange(0, n, step, dtype=int)
            tick_labels = plot_df["period"].iloc[tick_pos].tolist()
        else:
            step = max(1, int(np.ceil(len(year_pos) / max_xticks)))
            tick_pos = np.asarray(year_pos[::step], dtype=int)
            tick_labels = [year_labels[i] for i in range(0, len(year_labels), step)]
    else:
        step = max(1, int(np.ceil(n / max_xticks)))
        tick_pos = np.arange(0, n, step, dtype=int)
        tick_labels = plot_df["period"].iloc[tick_pos].tolist()

    if fig_width > 0:
        fig_w = float(fig_width)
    else:
        fig_w = float(np.clip(n * float(fig_width_per_point), float(fig_min_width), float(fig_max_width)))

    fig = plt.figure(figsize=(fig_w, float(fig_height)))
    ax = fig.add_subplot(1, 1, 1)
    y = plot_df[y_key].to_numpy(dtype=float, copy=False)
    smooth_w = max(1, int(smooth_window))
    if smooth_w > 1:
        y = pd.Series(y).rolling(window=smooth_w, center=True, min_periods=1).mean().to_numpy()

    ax.plot(
        x,
        y,
        marker="o",
        markersize=float(marker_size),
        markevery=max(1, int(marker_every)),
        linewidth=float(line_width),
    )
    ax.set_title(title)
    ax.set_xlabel("period")
    ax.set_ylabel(y_key)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch generate survival curves for train+test restaurant ids.")
    p.add_argument("--train-ids-file", type=str, default="logs_ids/split_train_all.txt")
    p.add_argument("--test-ids-file", type=str, default="logs_ids/split_test_all.txt")
    p.add_argument("--out-dir", type=str, default="plots_survival_curves")

    p.add_argument("--model", type=str, choices=["student", "teacher"], default="student")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="", help="可选：survival_hsk/data 目录（默认自动定位）")
    p.add_argument("--time-step", type=str, choices=["year", "half", "quarter"], default="quarter")
    p.add_argument("--min-year", type=int, default=None)
    p.add_argument("--max-year", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--text-dim", type=int, default=768)
    p.add_argument("--img-dim", type=int, default=512)
    p.add_argument("--feature-dim", type=int, default=8)
    p.add_argument("--max-offset-years", type=int, default=10)
    p.add_argument("--last-window-years", type=int, default=2)
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")

    p.add_argument("--max-restaurants", type=int, default=None, help="调试用：最多处理多少家店")
    p.add_argument("--skip-existing", action="store_true", help="若输出文件已存在则跳过（默认开启）")

    # 画图参数（与 predict_survival_curve 对齐）
    p.add_argument("--plot", action="store_true", help="输出 PNG 折线图")
    p.add_argument("--plot-y", type=str, choices=["survival", "closed"], default="survival")
    p.add_argument("--max-xticks", type=int, default=24)
    p.add_argument("--xtick-year-only", action="store_true")
    p.add_argument("--plot-x", type=str, choices=["period", "year"], default="period")
    p.add_argument("--year-reducer", type=str, choices=["mean", "first", "last"], default="mean")
    p.add_argument("--line-width", type=float, default=1.5)
    p.add_argument("--marker-size", type=float, default=2.0)
    p.add_argument("--marker-every", type=int, default=1)
    p.add_argument("--plot-smooth-window", type=int, default=1)
    p.add_argument("--fig-width", type=float, default=0.0)
    p.add_argument("--fig-height", type=float, default=4.0)
    p.add_argument("--fig-width-per-point", type=float, default=0.22)
    p.add_argument("--fig-min-width", type=float, default=8.0)
    p.add_argument("--fig-max-width", type=float, default=14.0)
    return p.parse_args()


def _predict_one(
    *,
    restaurant_id: str,
    model: torch.nn.Module,
    data_dir: Optional[Path],
    time_step: TimeStep,
    min_year: Optional[int],
    max_year: Optional[int],
    text_dim: int,
    img_dim: int,
    feature_dim: int,
    last_window_years: int,
    device: torch.device,
) -> pd.DataFrame:
    region_encoding = _load_restaurant_region(restaurant_id, data_dir=data_dir)
    review_df_raw = _load_reviews_for_restaurant(restaurant_id, data_dir=data_dir)
    review_df = prepare_reviews_clean(review_df_raw, time_step=time_step)
    review_df = review_df.sort_values("review_date", kind="mergesort")

    max_reviews = 128
    if len(review_df) > max_reviews:
        review_df = review_df.sample(n=max_reviews, random_state=42).sort_values("review_date", kind="mergesort")

    review_ids = review_df["review_id"].astype(str).tolist()
    bert_df = _load_bert_vectors_for_review_ids(review_ids, data_dir=data_dir, text_dim=text_dim)
    bert_map = build_bert_vector_map(bert_df, text_dim=text_dim)

    img_map = None
    img_df = _load_img_vectors_for_review_ids(review_ids, data_dir=data_dir, img_dim=img_dim)
    if img_df is not None and not img_df.empty:
        img_map = build_img_vector_map(img_df, img_dim=img_dim)

    cache = build_restaurant_review_cache(
        review_df,
        bert_map,
        img_map,
        max_reviews=max_reviews,
        text_dim=text_dim,
        img_dim=img_dim,
        feature_dim=feature_dim,
    )
    reviews = cache.get(str(restaurant_id))
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
    y0 = int(min_year) if min_year is not None else int(min_year_auto)
    y1 = int(max_year) if max_year is not None else int(max_year_auto)
    if y0 > y1:
        raise ValueError(f"min-year ({y0}) > max-year ({y1})")

    rows_out: List[Dict[str, object]] = []
    with torch.no_grad():
        for period in _iter_periods(y0, y1, time_step=time_step):
            batch = _build_batch_for_period(
                period_idx=period.idx,
                reviews=reviews,
                region_encoding=region_encoding,
                time_step=time_step,
                last_window_years=last_window_years,
                device=device,
            )
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

    return pd.DataFrame(rows_out)


def main() -> None:
    args = parse_args()
    time_step: TimeStep = args.time_step

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    data_dir = Path(args.data_dir) if args.data_dir else None
    out_dir = Path(args.out_dir)

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

    split_files = [
        ("train", Path(args.train_ids_file)),
        ("test", Path(args.test_ids_file)),
    ]

    y_key = "prob_survival" if args.plot_y == "survival" else "prob_closed"
    processed = 0
    failed = 0

    for split, ids_path in split_files:
        ids = _read_ids_file(ids_path)
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        for rid in ids:
            processed += 1
            if args.max_restaurants is not None and processed > int(args.max_restaurants):
                break

            out_csv = split_out / f"survival_curve_{rid}.csv"
            out_png = split_out / f"survival_curve_{rid}.png"

            if args.skip_existing and out_csv.exists() and (not args.plot or out_png.exists()):
                continue

            try:
                df_out = _predict_one(
                    restaurant_id=str(rid),
                    model=model,
                    data_dir=data_dir,
                    time_step=time_step,
                    min_year=args.min_year,
                    max_year=args.max_year,
                    text_dim=int(args.text_dim),
                    img_dim=int(args.img_dim),
                    feature_dim=int(args.feature_dim),
                    last_window_years=int(args.last_window_years),
                    device=device,
                )
                df_out.to_csv(out_csv, index=False)

                if args.plot:
                    _plot_curve(
                        df_out,
                        out_png=out_png,
                        title=f"{y_key} ({args.model}) - restaurant_id={rid}",
                        time_step=time_step,
                        y_key=y_key,
                        max_xticks=int(args.max_xticks),
                        xtick_year_only=bool(args.xtick_year_only),
                        plot_x=str(args.plot_x),
                        year_reducer=str(args.year_reducer),
                        line_width=float(args.line_width),
                        marker_size=float(args.marker_size),
                        marker_every=int(args.marker_every),
                        smooth_window=int(args.plot_smooth_window),
                        fig_width=float(args.fig_width),
                        fig_height=float(args.fig_height),
                        fig_width_per_point=float(args.fig_width_per_point),
                        fig_min_width=float(args.fig_min_width),
                        fig_max_width=float(args.fig_max_width),
                    )
            except Exception as exc:
                failed += 1
                # 打印少量错误，避免刷屏
                if failed <= 20:
                    print(f"[WARN] failed split={split} restaurant_id={rid}: {exc}")

        if args.max_restaurants is not None and processed > int(args.max_restaurants):
            break

    print(f"DONE processed={processed} failed={failed} out_dir={out_dir}")


if __name__ == "__main__":
    main()

