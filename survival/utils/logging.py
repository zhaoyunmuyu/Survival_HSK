"""日志配置工具

功能：
- 同时输出到日志文件与控制台；
- 简化训练脚本中的日志初始化；
- 返回日志路径与 logger，便于上层统一管理。
"""

import logging
import os
from typing import Tuple


def setup_logging(log_dir: str, filename: str) -> Tuple[str, logging.Logger]:
    """配置日志系统，写入磁盘并同步打印到标准输出。"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
        force=True,
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("").addHandler(console_handler)

    return log_path, logging.getLogger(__name__)
