from __future__ import annotations

"""
一键运行当前项目中所有的数据预处理脚本。

包含步骤：
1. 生成餐厅基础信息表（restaurant_base）；
2. 生成每条评论级别的清洗表（reviews_clean）；
3. 生成按餐厅聚合的评论统计特征表（reviews_agg_by_restaurant）；
4. （可选）对全部评论文本进行 BERT 向量化（review_bert_emb）。

特性：
- 不修改 openrice 目录下的任何原始数据；
- 可通过环境变量/命令行指定原始 openrice 数据目录；
- BERT 向量化过程会实时打印进度。
"""

from pathlib import Path
from typing import Optional
import argparse
import os

import pandas as pd

from .build_restaurant_base import (
    get_project_root,
    load_raw_restaurant_table,
    preprocess_restaurant_table,
)
from .build_reviews_clean import clean_reviews
from .build_reviews_agg import aggregate_reviews_to_restaurant
from .build_review_bert_emb import build_bert_embeddings


def parse_args() -> argparse.Namespace:
    """
    解析一键预处理脚本的命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="一键运行 restaurant_base / reviews_clean / reviews_agg_by_restaurant / review_bert_emb 的预处理脚本。"
    )
    parser.add_argument(
        "--openrice-dir",
        type=str,
        default=None,
        help="原始 openrice 数据目录（包含整合 CSV 等），例如 /data/openrice；"
             "若不指定，则默认使用当前工程目录下的 openrice/。",
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help="跳过 BERT 文本向量化步骤。",
    )
    parser.add_argument(
        "--reviews-clean-chunk-size",
        type=int,
        default=100_000,
        help="生成 reviews_clean 时读取 CSV 的分块大小。",
    )
    parser.add_argument(
        "--reviews-agg-chunk-size",
        type=int,
        default=100_000,
        help="生成 reviews_agg_by_restaurant 时读取 CSV 的分块大小。",
    )
    parser.add_argument(
        "--bert-max-rows",
        type=int,
        default=None,
        help="BERT 向量化最多处理的原始评论行数（调试用），为 None 时处理全部。",
    )
    parser.add_argument(
        "--bert-chunk-size",
        type=int,
        default=50_000,
        help="BERT 向量化读取 CSV 的分块大小。",
    )
    parser.add_argument(
        "--bert-batch-size",
        type=int,
        default=32,
        help="BERT 编码时的 batch 大小。",
    )
    parser.add_argument(
        "--bert-max-length",
        type=int,
        default=128,
        help="BERT 编码时的最大 token 长度（过长文本将被截断）。",
    )
    parser.add_argument(
        "--bert-total-rows-hint",
        type=int,
        default=None,
        help="（可选）总评论行数，用于显示 BERT 进度百分比；不指定则只显示已处理数量。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="预处理结果输出的根目录。"
             "如果不指定，则若有 openrice-dir 则使用 <openrice-dir>/preprocessed；"
             "否则默认为 <project-root>/survival_hsk/data。",
    )
    return parser.parse_args()


def run_restaurant_base(openrice_dir: Optional[str], output_dir: Path) -> None:
    """
    运行餐厅基础信息的预处理脚本。
    """
    root = get_project_root()
    print("=== 步骤 1/4: 生成 restaurant_base ===")
    print(f"项目根目录: {root}")
    if openrice_dir:
        print(f"使用自定义 openrice 目录: {openrice_dir}")

    df_raw = load_raw_restaurant_table(root)
    print(f"原始餐厅表形状: {df_raw.shape}")

    df_proc = preprocess_restaurant_table(df_raw)
    print(f"预处理后餐厅表形状: {df_proc.shape}")

    parquet_path = output_dir / "restaurant_base.parquet"
    csv_path = output_dir / "restaurant_base.csv"

    print(f"保存 restaurant_base Parquet 到: {parquet_path}")
    df_proc.to_parquet(parquet_path, index=False)

    print(f"保存 restaurant_base CSV 到: {csv_path}")
    df_proc.to_csv(csv_path, index=False)


def run_reviews_clean(openrice_dir: Optional[str], chunk_size: int, output_dir: Path) -> None:
    """
    运行评论清洗脚本（每条评论粒度）。
    """
    print("=== 步骤 2/4: 生成 reviews_clean ===")
    if openrice_dir:
        print(f"使用自定义 openrice 目录: {openrice_dir}")

    clean_reviews(
        max_rows=None,
        chunk_size=chunk_size,
        output_dir=output_dir,
    )


def run_reviews_agg(openrice_dir: Optional[str], chunk_size: int, output_dir: Path) -> None:
    """
    运行评论聚合脚本（按餐厅粒度）。
    """
    print("=== 步骤 3/4: 生成 reviews_agg_by_restaurant ===")
    if openrice_dir:
        print(f"使用自定义 openrice 目录: {openrice_dir}")

    df_agg = aggregate_reviews_to_restaurant(
        max_rows=None,
        chunk_size=chunk_size,
    )
    print(f"评论聚合表形状: {df_agg.shape}")

    parquet_path = output_dir / "reviews_agg_by_restaurant.parquet"
    csv_path = output_dir / "reviews_agg_by_restaurant.csv"

    print(f"保存 reviews_agg_by_restaurant Parquet 到: {parquet_path}")
    df_agg.to_parquet(parquet_path, index=False)

    print(f"保存 reviews_agg_by_restaurant CSV 到: {csv_path}")
    df_agg.to_csv(csv_path, index=False)


def run_bert_embeddings(
    max_rows: Optional[int],
    chunk_size: int,
    batch_size: int,
    max_length: int,
    total_rows_hint: Optional[int],
) -> None:
    """
    运行 BERT 文本向量化脚本。
    """
    print("=== 步骤 4/4: 生成 review_bert_emb（BERT 文本向量） ===")
    build_bert_embeddings(
        max_rows=max_rows,
        chunk_size=chunk_size,
        batch_size=batch_size,
        max_length=max_length,
        total_rows_hint=total_rows_hint,
    )


def main() -> None:
    """
    一键运行所有预处理步骤。
    """
    args = parse_args()

    # 如果指定了 openrice 目录，通过环境变量传递给各个子脚本
    if args.openrice_dir:
        os.environ["OPENRICE_DIR"] = args.openrice_dir

    root = get_project_root()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.openrice_dir:
        # 若指定 openrice 目录，默认将结果保存到 <openrice-dir>/preprocessed
        output_dir = Path(args.openrice_dir) / "preprocessed"
    else:
        output_dir = root / "survival_hsk" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 餐厅基础信息
    run_restaurant_base(args.openrice_dir, output_dir=output_dir)

    # 2) 评论清洗（每条评论）
    run_reviews_clean(
        args.openrice_dir,
        chunk_size=args.reviews_clean_chunk_size,
        output_dir=output_dir,
    )

    # 3) 评论聚合（餐厅粒度）
    run_reviews_agg(
        args.openrice_dir,
        chunk_size=args.reviews_agg_chunk_size,
        output_dir=output_dir,
    )

    # 4) BERT 向量化（可选）
    if args.skip_bert:
        print("已选择跳过 BERT 文本向量化步骤。")
    else:
        run_bert_embeddings(
            max_rows=args.bert_max_rows,
            chunk_size=args.bert_chunk_size,
            batch_size=args.bert_batch_size,
            max_length=args.bert_max_length,
            total_rows_hint=args.bert_total_rows_hint,
        )

    print("=== 所有预处理步骤已完成 ===")


if __name__ == "__main__":
    main()
