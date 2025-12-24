from __future__ import annotations

from pathlib import Path
import os
import re
from typing import Optional

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

    # Fallback for expected layout: <root>/survival_new_data/preprocess/...
    return current.parents[2]


def load_raw_restaurant_table(root: Optional[Path] = None) -> pd.DataFrame:
    """
    加载带有生存信息的整合餐厅表。

    读取路径：
        openrice/时间标签/整合餐厅信息_rating_survive_fillna_refreshed.csv

    注意：不会修改原始文件，所有处理都在内存中完成。
    """
    # 优先从环境变量 OPENRICE_DIR 中读取原始 openrice 数据目录，
    # 例如：OPENRICE_DIR=/data/openrice
    openrice_dir_env = os.environ.get("OPENRICE_DIR")
    if openrice_dir_env:
        openrice_dir = Path(openrice_dir_env)
        src = (
            openrice_dir
            / "时间标签"
            / "整合餐厅信息_rating_survive_fillna_refreshed.csv"
        )
    else:
        if root is None:
            root = get_project_root()
        src = (
            root
            / "openrice"
            / "时间标签"
            / "整合餐厅信息_rating_survive_fillna_refreshed.csv"
        )
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    df = pd.read_csv(src)
    return df


_RATING_RE = re.compile(r"([0-9]+(?:\\.[0-9]+)?)")


def parse_rating_to_float(value):
    """
    将类似 '4星' / '4.5星' 的评分字符串解析为浮点数（4.0 / 4.5）。
    解析失败或为空值时返回 pandas.NA。
    """
    if pd.isna(value):
        return pd.NA
    text = str(value)
    match = _RATING_RE.search(text)
    if not match:
        return pd.NA
    try:
        return float(match.group(1))
    except ValueError:
        return pd.NA


def preprocess_restaurant_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    对餐厅粒度数据做基础清洗与简单特征构造。

    主要步骤（不改动原始 CSV，仅在副本上操作）：
    - 删除明显的索引列（如 'Unnamed: 0'）； 
    - 将 `is_open` 规范为数值类型（0/1，可空整数）； 
    - 从 `restaurant_rating` 解析出数值评分列 `restaurant_rating_value`； 
    - 构造简单运营时长特征 `operation_years`； 
    - 为生存年份字段增加缺失指示列。

    除去被删除的索引列外，原始字段都会保留。
    """
    df_proc = df.copy()

    # 删除常见的自动生成索引列（如果存在）
    if "Unnamed: 0" in df_proc.columns:
        df_proc = df_proc.drop(columns=["Unnamed: 0"])

    # 尽量将 is_open 转为整数标签（0/1，可空）
    if "is_open" in df_proc.columns:
        df_proc["is_open"] = (
            df_proc["is_open"]
            .astype("float")
            .round()
            .astype("Int64")
        )

    # 将人类可读的评分文本解析为数值
    if "restaurant_rating" in df_proc.columns:
        df_proc["restaurant_rating_value"] = df_proc["restaurant_rating"].map(
            parse_rating_to_float
        )

    # 按年份粗略计算运营时长（后续如有需要可以进一步细化）
    if {"operation_early_year", "operation_latest_year"}.issubset(df_proc.columns):
        years = df_proc["operation_latest_year"] - df_proc["operation_early_year"]
        df_proc["operation_years"] = years
        df_proc["operation_early_year_missing"] = (
            df_proc["operation_early_year"].isna().astype("Int8")
        )
        df_proc["operation_latest_year_missing"] = (
            df_proc["operation_latest_year"].isna().astype("Int8")
        )

    return df_proc


def get_output_paths(root: Optional[Path] = None) -> dict:
    """
    计算餐厅处理结果的输出路径。
    """
    if root is None:
        root = get_project_root()

    data_dir = root / "survival_hsk" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return {
        "parquet": data_dir / "restaurant_base.parquet",
        "csv": data_dir / "restaurant_base.csv",
    }


def main() -> None:
    """
    脚本入口：
    - 加载原始整合餐厅表；
    - 执行预处理；
    - 将结果以 Parquet 和 CSV 格式保存到 survival_hsk/data。
    """
    root = get_project_root()
    print(f"Project root: {root}")

    print("Loading raw restaurant table...")
    df_raw = load_raw_restaurant_table(root)
    print(f"Raw shape: {df_raw.shape}")

    print("Preprocessing restaurant table...")
    df_proc = preprocess_restaurant_table(df_raw)
    print(f"Processed shape: {df_proc.shape}")

    paths = get_output_paths(root)

    print(f"Saving Parquet to: {paths['parquet']}")
    df_proc.to_parquet(paths["parquet"], index=False)

    print(f"Saving CSV to: {paths['csv']}")
    df_proc.to_csv(paths["csv"], index=False)

    print("Done.")


if __name__ == "__main__":
    main()
