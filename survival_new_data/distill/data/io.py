from __future__ import annotations

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """返回项目根目录（包含 `openrice/` 的文件夹）。"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "openrice").is_dir():
            return parent
    return current.parents[3]


def get_default_data_dir(root: Optional[Path] = None) -> Path:
    """返回 `survival_hsk/data` 默认目录。"""
    root = root or get_project_root()
    return root / "survival_hsk" / "data"


def resolve_data_path(filename: str, *, data_dir: Optional[Path] = None) -> Path:
    """解析 `survival_hsk/data/<filename>` 的实际路径。"""
    base = data_dir or get_default_data_dir()
    return Path(base) / filename

