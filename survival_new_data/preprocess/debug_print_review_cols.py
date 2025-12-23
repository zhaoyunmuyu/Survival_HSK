from __future__ import annotations

"""
调试脚本：打印整合评论 CSV 的列名，方便确认字段命名。

只做一次性检查使用，不参与正式预处理流程。
"""

from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    """
    返回项目根目录（包含 `openrice/` 的文件夹）。
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "openrice").is_dir():
            return parent
    return current.parents[2]


def get_reviews_source_path(root: Path) -> Path:
    """
    返回整合评论原始 CSV 的路径。
    """
    return root / "openrice" / "整合餐厅评论信息.csv"


def main() -> None:
    root = get_project_root()
    src = get_reviews_source_path(root)
    print(f"项目根目录: {root}")
    print(f"评论原始文件: {src}")

    df_head = pd.read_csv(src, nrows=0)
    print("列名：")
    for col in df_head.columns:
        print(col)


if __name__ == "__main__":
    main()

