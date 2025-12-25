"""使用学生模型按年份预测单个餐厅的“仍营业概率”并绘制折线图。

用法示例：
  python -m survival_kd.scripts.predict_student_curve ^
      --restaurant-id 123456 ^
      --checkpoint checkpoints_kd/student_best.pt ^
      --hidden-dim 512

说明：
- 依赖训练好的学生模型（BiLSTMStudent），默认从 checkpoints_kd/student_best.pt 加载；
- 自动按该餐厅评论出现的年份范围逐年预测仍营业概率（is_open），并保存折线图 PNG；
- 注意：本项目的 `is_open` 通常表示“2019 年是否仍在营业”（标签年），不是逐年生存/死亡标签；
  因此这里的“按年份”本质是改变 reference_year/last2_mask 的构造方式，更适合做相对比较与趋势观察。
  若你要对齐 2019 标签做评估，建议加：`--min-year 2019 --max-year 2019 --time-shift-years 0`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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


def _load_restaurant_row(restaurant_id: str) -> Dict:
    rest_df = _read_parquet_df(_resolve_data_file_local("restaurant_data.parquet"))
    rest_df["restaurant_id"] = rest_df["restaurant_id"].astype(str)
    row_df = rest_df[rest_df["restaurant_id"] == restaurant_id]
    if row_df.empty:
        raise KeyError(f"Restaurant id {restaurant_id!r} not found in restaurant_data.parquet")
    return row_df.iloc[0].to_dict()


def _load_restaurant_review_df(restaurant_id: str) -> pd.DataFrame:
    review_df = _read_parquet_df(_resolve_data_file_local("review_data.parquet"))
    review_df["restaurant_id"] = review_df["restaurant_id"].astype(str)
    review_df = review_df[review_df["restaurant_id"] == restaurant_id]
    if review_df.empty:
        raise RuntimeError(f"No reviews found for restaurant id {restaurant_id!r}")
    return prepare_review_dataframe(review_df)


def _load_single_restaurant_context(
    restaurant_id: str,
) -> Tuple[
    Dict,
    pd.DataFrame,
    Dict[str, torch.Tensor],
    Dict[str, Dict[int, torch.Tensor]],
    torch.Tensor,
    str | None,
]:
    row = _load_restaurant_row(restaurant_id)
    review_df = _load_restaurant_review_df(restaurant_id)

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
    return row, review_df, reviews, macro_data, macro_default, region_key


def _build_ordered_review_df_for_debug(review_df: pd.DataFrame, max_reviews: int = 128) -> pd.DataFrame:
    """复现 build_restaurant_review_cache 的采样/排序逻辑，用于调试可解释性输出。"""
    df = review_df.copy()
    if len(df) > max_reviews:
        df = df.sample(n=max_reviews, random_state=42)
    df = df.sort_values("review_date", kind="mergesort").reset_index(drop=True)
    return df


def _safe_str(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def _photo_count(photo_id_field: object) -> int:
    s = _safe_str(photo_id_field).strip()
    if not s:
        return 0
    return len([p for p in (p.strip() for p in s.split(";")) if p])


def _safe_print(line: str) -> None:
    """避免 Windows 控制台编码（如 gbk）导致的 UnicodeEncodeError。"""
    try:
        print(line)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(line.encode(enc, errors="replace").decode(enc, errors="replace"))


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

    # 防止“未来评论泄露”：只允许使用 <= 当前评估 year 的历史评论
    if history_mode == "last2":
        last2 = (years_tensor <= year) & (years_tensor >= (year - 1)) & (years_tensor > 0)
    elif history_mode == "all":
        last2 = (years_tensor <= year) & (years_tensor > 0)
    elif history_mode == "years":
        win = max(1, int(history_years))
        last2 = (years_tensor <= year) & (years_tensor >= (year - (win - 1))) & (years_tensor > 0)
    else:
        raise ValueError(f"Unknown history_mode: {history_mode!r}")

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
        description="Predict yearly open probability for one restaurant using student model."
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
        "--history",
        type=str,
        choices=["all", "last2", "years"],
        default="all",
        help="构造 last2_mask（学生模型保留的历史评论范围）：all=截至该年的全部历史；last2=仅当年+前一年；years=最近N年",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=2,
        help="当 --history=years 时生效：使用最近 N 年（含当年）",
    )
    parser.add_argument(
        "--plot-time-step",
        type=str,
        choices=["year", "half-year", "quarter"],
        default="year",
        help="绘图时横坐标步长，仅影响插值后的曲线，不改变模型按年预测的结果",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="仅导出/打印每一年实际喂给模型的输入明细，不加载 checkpoint、不计算概率",
    )
    parser.add_argument(
        "--debug-dump",
        type=str,
        default="",
        help="若非空，将每一年推理时使用的关键输入明细导出为 JSONL 文件（每行一个年份）",
    )
    parser.add_argument(
        "--debug-max-rows",
        type=int,
        default=12,
        help="debug 打印时每年最多展示的 last2 评论条数（其余只计数；0 表示不打印明细）",
    )
    parser.add_argument(
        "--debug-text-chars",
        type=int,
        default=120,
        help="debug 打印/导出时每条评论文本截断长度",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.time_shift_years:
        print(
            "[Note] --time-shift-years 仅用于输出/绘图的横轴平移；"
            "模型输入仍使用 review_year 作为 reference_year，并按该年选择 last2 评论与 macro 年份。"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student: BiLSTMStudent | None = None
    if args.skip_model or (args.debug_dump and not os.path.exists(args.checkpoint)):
        if not os.path.exists(args.checkpoint):
            print(f"[Debug] Student checkpoint not found: {args.checkpoint}; run in --skip-model mode.")
        else:
            print("[Debug] --skip-model enabled; skip loading student checkpoint.")
    else:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Student checkpoint not found: {args.checkpoint}")
        student = BiLSTMStudent(d_model=args.hidden_dim)
        state = torch.load(args.checkpoint, map_location="cpu")
        student.load_state_dict(state.get("state", state))
        student.to(device)
        student.eval()

    calibrator: LogitCalibrator | None = None
    if args.calibrator_checkpoint and student is not None:
        if not os.path.exists(args.calibrator_checkpoint):
            raise FileNotFoundError(f"Calibrator checkpoint not found: {args.calibrator_checkpoint}")
        cal_state = torch.load(args.calibrator_checkpoint, map_location="cpu")
        calibrator = LogitCalibrator()
        calibrator.load_state_dict(cal_state.get("state", cal_state))
        calibrator.to(device)
        calibrator.eval()
    elif args.calibrator_checkpoint and student is None:
        print("[Debug] --skip-model enabled; ignore --calibrator-checkpoint.")

    if student is None and args.skip_model:
        # Debug-only：只需要 restaurant_data + review_data，就能复现 last2 的选取逻辑
        row = _load_restaurant_row(args.restaurant_id)
        review_df = _load_restaurant_review_df(args.restaurant_id)
        reviews = {}
        macro_data = {}
        macro_default = torch.zeros(62, dtype=torch.float32)
        region_key = row.get("region_code") if isinstance(row.get("region_code"), str) else None
        years_tensor = torch.tensor(review_df["review_year"].astype(int).to_numpy(), dtype=torch.long)
    else:
        row, review_df, reviews, macro_data, macro_default, region_key = _load_single_restaurant_context(args.restaurant_id)
        years_tensor = reviews["years"].long()

    years: List[int] = []
    probs: List[float] = []
    last2_counts: List[int] = []
    debug_rows: List[dict] = []

    ordered_review_df = _build_ordered_review_df_for_debug(review_df)
    ordered_review_ids = ordered_review_df["review_id"].astype(str).tolist()
    ordered_review_dates = ordered_review_df["review_date"].tolist()
    ordered_review_years = ordered_review_df["review_year"].tolist() if "review_year" in ordered_review_df.columns else []
    ordered_review_texts = ordered_review_df["review_text"].astype("string").fillna("").tolist()

    # 统一 years_tensor 与 ordered_review_df 对齐（debug-only 模式不依赖 cache）
    if student is None and args.skip_model:
        max_reviews = 128
        padded_years = (ordered_review_years + [0] * max(0, max_reviews - len(ordered_review_years)))[:max_reviews]
        years_tensor = torch.tensor(padded_years, dtype=torch.long)

    year_values = sorted({int(y) for y in years_tensor.view(-1).tolist() if int(y) > 0})
    if not year_values:
        raise RuntimeError(f"No valid review years found for restaurant id {args.restaurant_id!r}")

    min_year = args.min_year if args.min_year is not None else year_values[0]
    max_year = args.max_year if args.max_year is not None else year_values[-1]
    if min_year > max_year:
        raise ValueError(f"min-year ({min_year}) should not be greater than max-year ({max_year})")

    with torch.no_grad():
        for year in range(min_year, max_year + 1):
            if student is None:
                if args.history == "last2":
                    last2_mask = (years_tensor <= year) & (years_tensor >= (year - 1)) & (years_tensor > 0)
                elif args.history == "all":
                    last2_mask = (years_tensor <= year) & (years_tensor > 0)
                else:
                    win = max(1, int(args.history_years))
                    last2_mask = (years_tensor <= year) & (years_tensor >= (year - (win - 1))) & (years_tensor > 0)
                last2_counts.append(int(last2_mask.sum().item()))
                prob = float("nan")
                selected_idx = np.where(last2_mask.cpu().view(-1).numpy().astype(bool))[0].tolist()
            else:
                batch = _build_batch_for_year(
                    year,
                    reviews,
                    macro_data,
                    macro_default,
                    region_key,
                    history_mode=args.history,
                    history_years=args.history_years,
                )
                last2_counts.append(int(batch["last2_mask"].sum().item()))
                for key, value in batch.items():
                    batch[key] = value.to(device)
                logits = student(batch)
                if calibrator is not None:
                    logits = calibrator(logits)
                prob = torch.sigmoid(logits).detach().cpu().item()
                selected_idx = np.where(batch["last2_mask"].detach().cpu().view(-1).bool().numpy())[0].tolist()
            years.append(year)
            probs.append(float(prob))

            if args.debug_dump:
                macro_available = bool(macro_data)
                region_macro = macro_data.get(region_key, {}) if (macro_available and region_key) else {}
                macro_hit: bool | None = (year in region_macro) if macro_available else None

                last2_reviews: List[dict] = []
                for idx in selected_idx:
                    if idx >= len(ordered_review_ids):
                        continue
                    txt = _safe_str(ordered_review_texts[idx]).replace("\r", " ").replace("\n", " ").strip()
                    if args.debug_text_chars and len(txt) > args.debug_text_chars:
                        txt = txt[: args.debug_text_chars] + "..."
                    dt = ordered_review_dates[idx]
                    dt_str = dt.isoformat() if hasattr(dt, "isoformat") else _safe_str(dt)
                    last2_reviews.append(
                        {
                            "idx": int(idx),
                            "review_id": ordered_review_ids[idx],
                            "review_date": dt_str,
                            "review_year": int(ordered_review_years[idx]) if idx < len(ordered_review_years) else None,
                            "n_photos": int(_photo_count(ordered_review_df.loc[idx, "review_photo_id"])),
                            "review_text": txt,
                        }
                    )

                debug_rows.append(
                    {
                        "restaurant_id": args.restaurant_id,
                        "region_code": region_key,
                        "reference_year": int(year),
                        "eval_year": int(year + args.time_shift_years),
                        "n_reviews_total": int(len(ordered_review_ids)),
                        "n_last2_reviews": int(last2_counts[-1]),
                        "history_mode": str(args.history),
                        "history_years": int(args.history_years),
                        "macro_hit": macro_hit,
                        "macro_available": bool(macro_available),
                        "pred_open_prob": float(prob),
                        "last2_reviews": last2_reviews,
                    }
                )

    print(f"Restaurant id: {args.restaurant_id}")
    name = row.get("name") or row.get("restaurant_name") or ""
    if name:
        print(f"Name: {name}")
    print("Yearly open probabilities (student model, by review year):")
    for review_year, prob, n_last2 in zip(years, probs, last2_counts):
        target_year = review_year + args.time_shift_years
        suffix = ""
        if n_last2 <= 0:
            suffix = " (no last2 reviews; model relies on macro/region only)"
        prob_str = f"{prob:.4f}" if np.isfinite(prob) else "N/A"
        print(f"- review year {review_year} -> target year {target_year}: {prob_str} | last2_reviews={n_last2}{suffix}")

        if args.debug_max_rows > 0 and n_last2 > 0:
            # 打印该年 last2 具体用到哪些评论（最多 debug_max_rows 条）
            years_np = years_tensor.cpu().view(-1).numpy()
            if args.history == "last2":
                last2_mask_dbg = (years_np <= review_year) & (years_np >= (review_year - 1)) & (years_np > 0)
            elif args.history == "all":
                last2_mask_dbg = (years_np <= review_year) & (years_np > 0)
            else:
                win = max(1, int(args.history_years))
                last2_mask_dbg = (years_np <= review_year) & (years_np >= (review_year - (win - 1))) & (years_np > 0)
            idxs = np.where(last2_mask_dbg)[0].tolist()
            shown = 0
            for idx in idxs:
                if idx >= len(ordered_review_ids):
                    continue
                txt = _safe_str(ordered_review_texts[idx]).replace("\r", " ").replace("\n", " ").strip()
                if args.debug_text_chars and len(txt) > args.debug_text_chars:
                    txt = txt[: args.debug_text_chars] + "..."
                dt = ordered_review_dates[idx]
                dt_str = dt.date().isoformat() if hasattr(dt, "date") else _safe_str(dt)
                print(
                    f"  - idx={idx} review_id={ordered_review_ids[idx]} date={dt_str} "
                    f"photos={_photo_count(ordered_review_df.loc[idx, 'review_photo_id'])}"
                )
                _safe_print(f"    text: {txt}")
                shown += 1
                if shown >= args.debug_max_rows:
                    remaining = max(0, n_last2 - shown)
                    if remaining:
                        print(f"  ... ({remaining} more last2 reviews omitted)")
                    break

    if args.debug_dump and debug_rows:
        out_path = args.debug_dump
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            for row_obj in debug_rows:
                handle.write(json.dumps(row_obj, ensure_ascii=False) + "\n")
        print(f"[Debug] Saved per-year input details to: {out_path}")

    if student is None:
        if args.debug_dump:
            return
        print("[Debug] --skip-model enabled; skip plotting.")
        return

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
    plt.ylabel("Open probability (is_open)")
    plt.title(f"Student model open prob - {args.restaurant_id}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved line chart to: {output_path}")


if __name__ == "__main__":
    main()
