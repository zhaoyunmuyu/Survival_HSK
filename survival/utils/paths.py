"""数据路径解析工具

优先级（从高到低）：
1) 环境变量 `SURVIVAL_DATA_ROOT` 指定的数据根目录；
2) 仓库根的同级目录 `junLi`（适配 /home/gvmuserthree/yzang_lab/junLi）；
3) 仓库根下 `data/raw/`；
4) 当前工作目录。

提供统一入口以便不同模块共用。
"""

from __future__ import annotations

import os
from pathlib import Path


def _repo_root_from(path: Path) -> Path:
    # 允许从任一文件定位仓库根（即包含 survival/ 的根目录）
    # 约定：本文件位于 <repo_root>/survival/utils/paths.py => parents[2]
    try:
        return path.resolve().parents[2]
    except Exception:
        return Path.cwd()


def get_data_root() -> Path:
    """解析数据根目录，返回 Path。

    - 若设置了 `SURVIVAL_DATA_ROOT` 且有效，优先返回；
    - 否则尝试 <repo_root>/../junLi；
    - 否则 <repo_root>/data/raw；
    - 否则 CWD。
    """
    env_path = os.getenv("SURVIVAL_DATA_ROOT")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    repo_root = _repo_root_from(Path(__file__))
    sibling = repo_root.parent / "junLi"
    if sibling.exists():
        return sibling

    raw = repo_root / "data" / "raw"
    if raw.exists():
        return raw

    return Path.cwd()


def resolve_data_file(filename: str) -> str:
    """在数据根目录下解析文件，若不存在则回退到旧策略。"""
    base = get_data_root()
    candidate = base / filename
    if candidate.exists():
        return str(candidate)

    # 兜底：兼容原路径策略
    repo_root = _repo_root_from(Path(__file__))
    raw_path = repo_root / "data" / "raw" / filename
    if raw_path.exists():
        return str(raw_path)
    cwd_path = Path.cwd() / filename
    if cwd_path.exists():
        return str(cwd_path)
    return str(candidate)  # 返回期望路径，供上层抛出友好错误


def resolve_data_dir(dirname: str) -> str:
    """在数据根目录下解析子目录，若不存在则回退到旧策略。"""
    base = get_data_root()
    candidate = base / dirname
    if candidate.exists():
        return str(candidate)
    repo_root = _repo_root_from(Path(__file__))
    legacy = repo_root / dirname
    if legacy.exists():
        return str(legacy)
    alt = Path.cwd() / dirname
    return str(alt if alt.exists() else candidate)
