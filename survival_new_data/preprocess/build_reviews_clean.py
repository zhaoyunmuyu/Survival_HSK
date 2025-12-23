from __future__ import annotations

"""
从整合评论表中构建「每条评论」粒度的清洗结果表。

目标：
- 保留每条评论的详细信息，方便后续分析 / 调试；
- 做尽量轻量的预处理（删除索引列、规范日期类型），不改变原始含义；
- 不修改 openrice 目录下的任何原始数据。

输出：
- survival_hsk/data/reviews_clean.parquet
- survival_hsk/data/reviews_clean.csv

说明：
- 输入文件体积较大（GB 级），脚本使用分块读取 + 流式写出；
- 提供 --max-rows 参数，便于先在小样本上调试，再全量运行。
"""

from pathlib import Path
from typing import Optional, Dict
import argparse
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_project_root() -> Path:
    """
    返回项目根目录（包含 `openrice/` 的文件夹）。
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "openrice").is_dir():
            return parent
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


def get_reviews_clean_output_paths(
    root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    返回清洗后评论结果的输出路径（Parquet / CSV）。
    """
    if output_dir is not None:
        data_dir = output_dir
    else:
        if root is None:
            root = get_project_root()
        data_dir = root / "survival_hsk" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

    return {
        "parquet": data_dir / "reviews_clean.parquet",
        "csv": data_dir / "reviews_clean.csv",
    }


def clean_reviews(
    max_rows: Optional[int] = None,
    chunk_size: int = 100_000,
    output_dir: Optional[Path] = None,
) -> None:
    """
    逐块读取整合评论 CSV，进行轻量清洗后写出 Parquet 和 CSV。

    处理逻辑：
    - 删除明显的索引列 `Unnamed: 0`（如果存在）；
    - 将 `review_date` 解析为日期并再格式化为 `YYYY-MM-DD` 字符串；
    - 其它字段保持原样，不做筛选或填补。
    """
    root = get_project_root()
    src = get_reviews_source_path(root)
    paths = get_reviews_clean_output_paths(root, output_dir=output_dir)

    if not src.exists():
        raise FileNotFoundError(f"找不到评论原始文件：{src}")

    print(f"评论原始文件: {src}")
    print(f"清洗后 Parquet 输出: {paths['parquet']}")
    print(f"清洗后 CSV 输出: {paths['csv']}")

    # 如果之前已存在输出文件，可以先删除，避免旧数据残留
    for p in paths.values():
        if p.exists():
            p.unlink()

    parquet_writer: Optional[pq.ParquetWriter] = None
    first_chunk = True
    rows_read = 0

    for chunk in pd.read_csv(src, chunksize=chunk_size):
        if max_rows is not None:
            remaining = max_rows - rows_read
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        rows_read += len(chunk)

        # 删除自动生成的索引列
        if "Unnamed: 0" in chunk.columns:
            chunk = chunk.drop(columns=["Unnamed: 0"])

        # 解析并规范 review_date
        if "review_date" in chunk.columns:
            dt = pd.to_datetime(chunk["review_date"], errors="coerce")
            chunk["review_date"] = dt.dt.strftime("%Y-%m-%d")

        # 写入 Parquet（流式追加）
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(paths["parquet"], table.schema)
        parquet_writer.write_table(table)

        # 写入 CSV（首块写表头，后续追加）
        mode = "w" if first_chunk else "a"
        header = first_chunk
        chunk.to_csv(paths["csv"], index=False, mode=mode, header=header)
        first_chunk = False

        print(f"已处理评论行数: {rows_read}")

    if parquet_writer is not None:
        parquet_writer.close()

    print("清洗完成。")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="从整合评论表生成每条评论粒度的清洗结果表。"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="仅用于调试：最多读取的评论行数，为 None 时处理全部。",
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
    脚本入口。
    """
    args = parse_args()
    clean_reviews(
        max_rows=args.max_rows,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
