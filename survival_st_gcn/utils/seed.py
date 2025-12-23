"""随机性控制与播种工具

功能：
- 设置环境变量，固定相关库随机性；
- 统一播种 Python/NumPy/PyTorch（含多 GPU）；
- 为 DataLoader 提供稳定的 `torch.Generator` 和 worker 播种函数；

注意：
- 为了更高的可复现性，默认关闭 cuDNN 的 benchmark，开启 deterministic；
- 某些算子可能因此退化为确定性实现，训练速度略降属正常现象。
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

from .constants import GLOBAL_SEED


def _configure_environment(seed: int = GLOBAL_SEED) -> None:
    """设置与确定性运行相关的环境变量。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    """统一播种 Python、NumPy、PyTorch（含 GPU）。"""
    _configure_environment(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_generator(offset: int = 0) -> torch.Generator:
    """创建绑定固定种子的 `torch.Generator`（用于 DataLoader）。

    参数 offset 用于给不同 DataLoader 分配不同但稳定的种子，
    避免它们之间随机序列相互干扰。
    """
    generator = torch.Generator()
    generator.manual_seed(GLOBAL_SEED + offset)
    return generator


def seed_worker(worker_id: int) -> None:
    """为 DataLoader 的 worker 进程播种，确保每个 worker 的随机性独立且可复现。"""
    worker_seed = GLOBAL_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)


# 兼容旧代码接口（部分 DataLoader 仍使用 `worker_init_fn` 参数）
def worker_init_fn(worker_id: int) -> None:  # pragma: no cover - 兼容函数
    seed_worker(worker_id)
