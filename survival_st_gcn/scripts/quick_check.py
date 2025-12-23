"""
最小化数据加载验证（不训练）。

目标：
- 以 data/raw 优先策略读取必要文件；
- 使用 PyArrow 直接读取 parquet（规避 pandas 对 pyarrow 版本要求）；
- 构建少量样本的 Dataset 与年度图兜底，打印若干维度与示例。

运行：
  python -m survival.scripts.quick_check
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from survival_st_gcn.data.reviews import prepare_review_dataframe, build_restaurant_review_cache
from survival_st_gcn.data.text_vectors import build_text_vector_map
from survival_st_gcn.data.macro import prepare_macro_data
from survival_st_gcn.data.datasets import RestaurantDataset
from survival_st_gcn.data.graphs import load_yearly_graphs
from survival_st_gcn.utils.paths import get_data_root


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pick(name: str) -> Path:
    data_root = get_data_root()
    p1 = (data_root / name) if isinstance(data_root, Path) else Path(data_root) / name
    return p1 if p1.exists() else (Path.cwd() / name)


def _read_head_as_df(path: Path, *, row_group: int = 0, limit: Optional[int] = 5000) -> pd.DataFrame:
    """优先仅读入首个 row group，减少内存占用；若不可用则全读并在 pandas 侧切片。"""
    pf = pq.ParquetFile(str(path))
    try:
        tab = pf.read_row_group(row_group)
    except Exception:
        tab = pf.read()  # 兜底全读
    df = tab.to_pandas()
    if limit is not None and len(df) > limit:
        df = df.head(limit)
    return df


def main() -> None:
    # 1) 读取基础表（只取少量）
    p_rest = _pick("restaurant_data.parquet")
    p_review = _pick("review_data.parquet")
    p_text = _pick("text_vectors.parquet")
    p_macro = _pick("normalized_macro_data.json")

    print("读取路径:")
    print("- restaurant:", p_rest)
    print("- review:", p_review)
    print("- text_vectors:", p_text)
    print("- macro:", p_macro)

    rest_df = _read_head_as_df(p_rest, limit=10000)
    rest_df["restaurant_id"] = rest_df["restaurant_id"].astype(str)
    print("restaurant_data shape:", rest_df.shape)

    review_df = _read_head_as_df(p_review, limit=50000)
    review_df["restaurant_id"] = review_df["restaurant_id"].astype(str)
    print("review_data shape:", review_df.shape)

    text_df = _read_head_as_df(p_text, limit=50000)
    if "text_vector" not in text_df.columns and "element" in text_df.columns:
        text_df = text_df.rename(columns={"element": "text_vector"})
    print("text_vectors shape:", text_df.shape)

    # 2) 子集抽样，保证 restaurant/review/text 对齐
    rest_ids = rest_df["restaurant_id"].dropna().astype(str).unique().tolist()[:200]
    review_df = review_df[review_df["restaurant_id"].isin(set(rest_ids))].copy()
    review_df = prepare_review_dataframe(review_df)
    print("filtered review_data shape:", review_df.shape)

    # 只保留有文本向量的 review_id
    text_ids = set(text_df["review_id"].astype(str))
    review_df = review_df[review_df["review_id"].astype(str).isin(text_ids)]
    print("aligned review_data shape:", review_df.shape)

    # 3) 文本向量映射与图片/结构化缓存
    tv_map = build_text_vector_map(text_df)
    cache = build_restaurant_review_cache(review_df, tv_map, img_feat_dirs=())

    # 4) 宏观数据
    macro_raw = json.loads(Path(p_macro).read_text(encoding="utf-8"))
    macro_data, macro_default = prepare_macro_data(macro_raw)

    # 5) 数据集构建与图数据兜底
    rest_small = rest_df[rest_df["restaurant_id"].isin(set(cache.keys()))].reset_index(drop=True)
    if rest_small.empty:
        # 至少保底一个样本，构造空白占位
        rest_small = rest_df.head(1).copy()
        rest_small.loc[:, "is_open"] = 0
    dataset = RestaurantDataset(rest_small, cache, macro_data, macro_default)
    print("dataset length:", len(dataset))
    item0 = dataset[0]
    print("sample keys:", sorted(item0.keys()))
    for k, v in item0.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}:", tuple(v.shape), v.dtype)

    # 6) 年度图兜底加载
    # 从餐厅ID构建列表传给兜底图，确保索引映射一致
    rest_ids_numeric = (
        pd.to_numeric(rest_small["restaurant_id"], errors="coerce").dropna().astype(int).drop_duplicates().tolist()
    )
    from survival_st_gcn.utils.paths import resolve_data_dir
    graphs = load_yearly_graphs(resolve_data_dir("graph_data/10_year_graphs"), restaurant_ids=rest_ids_numeric)
    print("yearly_graphs count:", len(graphs))
    g0 = graphs[0]
    print("graph0 nodes:", int(getattr(g0, "total_rests", getattr(getattr(g0, "x", np.zeros((0,))), "shape", (0,))[0])))


if __name__ == "__main__":
    main()
