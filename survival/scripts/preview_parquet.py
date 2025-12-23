"""
Parquet 数据快速预览工具。

从 `data/raw` 目录中读取项目约定的几个 parquet 文件，各展示少量样例
与简要信息，便于快速核对字段与数据内容。

用法：
  python -m survival.scripts.preview_parquet
  # 或者
  python survival/scripts/preview_parquet.py
"""

# from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    # 可选导入：仅用于统计文本向量维度长度
    from survival.data.text_vectors import _coerce_vector
except Exception:  # pragma: no cover - best-effort import for preview utility
    _coerce_vector = None  # type: ignore


def _read_parquet(path: Path) -> Optional[pd.DataFrame]:
    """读取 parquet（尝试多种引擎），失败返回 None。"""
    for engine in ("pyarrow", "fastparquet", None):
        try:
            return pd.read_parquet(path, engine=engine)
        except Exception:
            continue
    return None


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "raw"
    cwd = Path.cwd()

    print(f"数据目录: {data_dir}")

    def pick(name: str) -> Path:
        # 优先 data/raw，其次工作目录
        p1 = data_dir / name
        return p1 if p1.exists() else (cwd / name)

    targets = {
        "restaurant_data": pick("restaurant_data.parquet"),
        "review_data": pick("review_data.parquet"),
        "text_vectors": pick("text_vectors.parquet"),
    }

    for name, path in targets.items():
        print("\n==>", name, "->", path)
        if not path.exists():
            print(f"- 缺失: {path}")
            continue
        df = _read_parquet(path)
        if df is None:
            print("- 读取失败（可能未安装/不兼容 pyarrow 或 fastparquet）")
            continue
        print(f"- 形状: {df.shape}")
        print(f"- 列: {list(df.columns)[:10]}{' ...' if len(df.columns) > 10 else ''}")

        # 按文件类型展示简洁示例
        if name == "restaurant_data":
            cols = [
                c for c in (
                    "restaurant_id",
                    "is_open",
                    "region_code",
                    "operation_latest_year",
                )
                if c in df.columns
            ]
            preview = df[cols].head(5) if cols else df.head(5)
            print(preview)
        elif name == "review_data":
            cols = [
                c for c in (
                    "restaurant_id",
                    "review_id",
                    "review_photo_id",
                    "review_date",
                    "user_location",
                )
                if c in df.columns
            ]
            preview = df[cols].head(5) if cols else df.head(5)
            print(preview)
        elif name == "text_vectors":
            if {"review_id", "text_vector"}.issubset(df.columns):
                rows = df[["review_id", "text_vector"]].head(5).to_dict("records")
                for i, row in enumerate(rows, 1):
                    rid = str(row["review_id"]) if row.get("review_id") is not None else "<NA>"
                    if _coerce_vector is not None:
                        try:
                            vec = _coerce_vector(row["text_vector"])  # type: ignore[arg-type]
                            dim = len(vec) if vec is not None else 0
                        except Exception:
                            dim = 0
                        print(f"  [{i}] review_id={rid} 向量维度={dim}")
                    else:
                        print(f"  [{i}] review_id={rid} (向量未解析)")
            else:
                print(df.head(5))


if __name__ == "__main__":
    main()
