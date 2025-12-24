from __future__ import annotations

"""
从整合评论表中按餐厅粒度构建评论聚合特征（不做 TF-IDF 向量化）。

注意：
- 不会修改 openrice 目录下的任何原始数据；
- 输出结果保存到 survival_hsk/data/reviews_agg_by_restaurant.(parquet/csv)；
- 当前脚本只生成数值型和时间型的统计特征，文本语义信息将由单独的 BERT 向量化脚本处理。
"""

from pathlib import Path
from typing import Dict, Optional, Any
import argparse
import math
import os

import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """
    返回项目根目录（包含 `openrice/` 的文件夹）。

    脚本可能位于多级子目录（例如 `survival_new_data/preprocess`），
    因此这里自下而上查找，直到找到包含 `openrice/` 的目录。
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "openrice").is_dir():
            return parent

    # 兜底策略：按预期目录结构回退两级
    return current.parents[2]


def get_reviews_source_path(root: Optional[Path] = None) -> Path:
    """
    返回整合评论原始 CSV 的路径。
    """
    # 优先从环境变量 OPENRICE_DIR 中读取原始 openrice 数据目录，
    # 例如：OPENRICE_DIR=/data/openrice
    openrice_dir_env = os.environ.get("OPENRICE_DIR")
    if openrice_dir_env:
        openrice_dir = Path(openrice_dir_env)
        return openrice_dir / "整合餐厅评论信息.csv"

    if root is None:
        root = get_project_root()
    return root / "openrice" / "整合餐厅评论信息.csv"


def get_reviews_output_paths(root: Optional[Path] = None) -> Dict[str, Path]:
    """
    返回评论聚合结果的输出路径（Parquet / CSV）。
    """
    if root is None:
        root = get_project_root()

    data_dir = root / "survival_hsk" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return {
        "parquet": data_dir / "reviews_agg_by_restaurant.parquet",
        "csv": data_dir / "reviews_agg_by_restaurant.csv",
    }


def aggregate_reviews_to_restaurant(
    max_rows: Optional[int] = None,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """
    从整合评论 CSV 中按餐厅聚合，得到餐厅粒度的评论统计特征。

    参数：
        max_rows: 仅用于调试，限制最多读取的评论行数；为 None 时读取全部。
        chunk_size: 分块读取 CSV 的块大小。
    """
    root = get_project_root()
    src = get_reviews_source_path(root)
    if not src.exists():
        raise FileNotFoundError(f"找不到评论原始文件：{src}")

    # 只读取后续会用到的列，减少 IO 和内存压力
    usecols = [
        "restaurant_id",
        "review_id",
        "review_date",
        "review_taste",
        "text_length",
        "text_emoji_num",
        "text_pos_emoji_num",
        "text_neu_emoji_num",
        "text_neg_emoji_num",
        "review_photo_num",
    ]

    agg: Dict[Any, Dict[str, Any]] = {}

    rows_read = 0
    for chunk in pd.read_csv(src, usecols=usecols, chunksize=chunk_size):
        if max_rows is not None:
            remaining = max_rows - rows_read
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        rows_read += len(chunk)

        # 解析日期
        chunk["review_date"] = pd.to_datetime(
            chunk["review_date"], errors="coerce"
        )

        for row in chunk.itertuples(index=False):
            restaurant_id = getattr(row, "restaurant_id")
            if pd.isna(restaurant_id):
                continue

            entry = agg.get(restaurant_id)
            if entry is None:
                entry = {
                    "n_reviews": 0,
                    "n_reviews_with_rating": 0,
                    "n_reviews_with_photo": 0,
                    "rating_sum": 0.0,
                    "rating_sq_sum": 0.0,
                    "rating_low_count": 0,
                    "rating_high_count": 0,
                    "review_length_sum": 0.0,
                    "review_length_sq_sum": 0.0,
                    "review_length_max": 0,
                    "emoji_num_sum": 0.0,
                    "pos_emoji_sum": 0.0,
                    "neu_emoji_sum": 0.0,
                    "neg_emoji_sum": 0.0,
                    "first_review_date": None,
                    "last_review_date": None,
                }
                agg[restaurant_id] = entry

            entry["n_reviews"] += 1

            rating = getattr(row, "review_taste")
            if not pd.isna(rating):
                try:
                    rating_val = float(rating)
                except ValueError:
                    rating_val = None
                if rating_val is not None:
                    entry["n_reviews_with_rating"] += 1
                    entry["rating_sum"] += rating_val
                    entry["rating_sq_sum"] += rating_val * rating_val
                    if rating_val <= 3.0:
                        entry["rating_low_count"] += 1
                    if rating_val >= 4.5:
                        entry["rating_high_count"] += 1

            photo_num = getattr(row, "review_photo_num")
            if not pd.isna(photo_num) and photo_num > 0:
                entry["n_reviews_with_photo"] += 1

            length = getattr(row, "text_length")
            if not pd.isna(length):
                try:
                    length_val = int(length)
                except ValueError:
                    length_val = None
                if length_val is not None:
                    entry["review_length_sum"] += length_val
                    entry["review_length_sq_sum"] += length_val * length_val
                    if length_val > entry["review_length_max"]:
                        entry["review_length_max"] = length_val

            emoji_num = getattr(row, "text_emoji_num")
            if not pd.isna(emoji_num):
                entry["emoji_num_sum"] += float(emoji_num)

            pos_emoji = getattr(row, "text_pos_emoji_num")
            if not pd.isna(pos_emoji):
                entry["pos_emoji_sum"] += float(pos_emoji)

            neu_emoji = getattr(row, "text_neu_emoji_num")
            if not pd.isna(neu_emoji):
                entry["neu_emoji_sum"] += float(neu_emoji)

            neg_emoji = getattr(row, "text_neg_emoji_num")
            if not pd.isna(neg_emoji):
                entry["neg_emoji_sum"] += float(neg_emoji)

            review_date = getattr(row, "review_date")
            if isinstance(review_date, pd.Timestamp):
                if entry["first_review_date"] is None or review_date < entry[
                    "first_review_date"
                ]:
                    entry["first_review_date"] = review_date
                if entry["last_review_date"] is None or review_date > entry[
                    "last_review_date"
                ]:
                    entry["last_review_date"] = review_date

    # 将聚合结果转为 DataFrame，并计算派生统计特征
    records = []
    for restaurant_id, e in agg.items():
        n = e["n_reviews"]
        n_rating = e["n_reviews_with_rating"]

        rec: Dict[str, Any] = {
            "restaurant_id": restaurant_id,
            "n_reviews": n,
            "n_reviews_with_rating": n_rating,
            "n_reviews_with_photo": e["n_reviews_with_photo"],
            "first_review_date": e["first_review_date"],
            "last_review_date": e["last_review_date"],
            "review_length_max": e["review_length_max"],
        }

        # 评分统计
        if n_rating > 0:
            mean_rating = e["rating_sum"] / n_rating
            var_rating = max(
                0.0,
                e["rating_sq_sum"] / n_rating - mean_rating * mean_rating,
            )
            rec["rating_mean"] = mean_rating
            rec["rating_std"] = math.sqrt(var_rating)
            rec["rating_low_ratio"] = e["rating_low_count"] / n_rating
            rec["rating_high_ratio"] = e["rating_high_count"] / n_rating
        else:
            rec["rating_mean"] = np.nan
            rec["rating_std"] = np.nan
            rec["rating_low_ratio"] = np.nan
            rec["rating_high_ratio"] = np.nan

        # 文本长度统计
        if n > 0 and e["review_length_sum"] > 0:
            mean_len = e["review_length_sum"] / n
            var_len = max(
                0.0,
                e["review_length_sq_sum"] / n - mean_len * mean_len,
            )
            rec["review_length_mean"] = mean_len
            rec["review_length_std"] = math.sqrt(var_len)
        else:
            rec["review_length_mean"] = np.nan
            rec["review_length_std"] = np.nan

        # emoji 统计
        if n > 0:
            rec["emoji_num_mean"] = e["emoji_num_sum"] / n
        else:
            rec["emoji_num_mean"] = np.nan

        total_emoji = e["emoji_num_sum"]
        if total_emoji > 0:
            rec["pos_emoji_ratio"] = e["pos_emoji_sum"] / total_emoji
            rec["neu_emoji_ratio"] = e["neu_emoji_sum"] / total_emoji
            rec["neg_emoji_ratio"] = e["neg_emoji_sum"] / total_emoji
        else:
            rec["pos_emoji_ratio"] = np.nan
            rec["neu_emoji_ratio"] = np.nan
            rec["neg_emoji_ratio"] = np.nan

        # 评论时间跨度（天）
        if (
            isinstance(e["first_review_date"], pd.Timestamp)
            and isinstance(e["last_review_date"], pd.Timestamp)
        ):
            delta = e["last_review_date"] - e["first_review_date"]
            rec["review_span_days"] = delta.days
        else:
            rec["review_span_days"] = np.nan

        records.append(rec)

    df_agg = pd.DataFrame.from_records(records)
    return df_agg


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="从整合评论表构建餐厅粒度的评论聚合特征（不含 TF-IDF 向量）。"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="仅用于调试：最多读取的评论行数，为 None 时读取全部。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="分块读取 CSV 的块大小。",
    )
    return parser.parse_args()


def main() -> None:
    """
    脚本入口：构建评论聚合特征并保存到 data 目录。
    """
    args = parse_args()

    root = get_project_root()
    print(f"项目根目录: {root}")

    print("开始按餐厅聚合评论特征...")
    df_agg = aggregate_reviews_to_restaurant(
        max_rows=args.max_rows,
        chunk_size=args.chunk_size,
    )
    print(f"餐厅聚合表形状: {df_agg.shape}")

    paths = get_reviews_output_paths(root)
    print(f"保存 Parquet 到: {paths['parquet']}")
    df_agg.to_parquet(paths["parquet"], index=False)

    print(f"保存 CSV 到: {paths['csv']}")
    df_agg.to_csv(paths["csv"], index=False)

    print("完成。")


if __name__ == "__main__":
    main()
